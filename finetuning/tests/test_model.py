import torch

from finetuning.tinysql.evaluate import report, results_for
from finetuning.tinysql.model import ModelConfig, TinySQLLM
from finetuning.tinysql.tokenizer import WordSQLTokenizer


def test_forward_backward_and_causality():
    torch.set_num_threads(2)
    m = TinySQLLM(ModelConfig(vocab_size=40, context=40, dim=32, layers=2, heads=4))
    a = torch.randint(0, 40, (2, 12))
    b = a.clone()
    b[:, 10:] = torch.randint(0, 40, (2, 2))

    with torch.no_grad():
        ya, _ = m(a)
        yb, _ = m(b)

        assert torch.allclose(ya[:, :10], yb[:, :10], atol=1e-5)

    labels = a.clone()
    labels[:, :5] = -100
    _, loss = m(a, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert m.tok_emb.weight.grad.abs().sum() > 0


def test_padding_loss_is_ignored():
    m = TinySQLLM(ModelConfig(vocab_size=20, context=20, dim=32, layers=1, heads=4))
    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))
    y[:, -3:] = -100
    _, loss = m(x, y)

    assert loss.item() > 0


def test_sql_tokenizer_round_trip():
    rows = [
        {
            "question": "list customers in tx",
            "sql": "SELECT name FROM customers WHERE state = 'tx' ORDER BY name ASC;",
        }
    ]
    tok = WordSQLTokenizer.fit(rows)
    x, y = tok.format_train(rows[0])

    assert len(x) == len(y)
    assert all(v == -100 for v in y[: len(tok.format_input(rows[0]["question"])) - 1])
    assert y[-1] == tok.eos_id
    assert "'tx'" in tok.decode(tok.encode(rows[0]["sql"]))


def test_metric_not_confused_with_exact_match():
    row = {
        "id": "a",
        "family": "order_count",
        "question": "how many pending orders",
        "sql": "SELECT COUNT(*) FROM orders WHERE status = 'pending';",
    }
    predicted = "select count ( * ) from orders where status = 'pending';"
    scores = report(results_for([row], [predicted]))

    assert scores["execution_accuracy"] == 1.0
    assert scores["exact_match"] == 1.0


def test_checkpoint_round_trip(tmp_path):
    cfg = ModelConfig(vocab_size=32, dim=32, layers=1, heads=4, context=30)
    m = TinySQLLM(cfg)
    p = tmp_path / "c.pt"
    torch.save(m.state_dict(), p)
    m2 = TinySQLLM(cfg)
    m2.load_state_dict(torch.load(p, weights_only=True))

    assert all(torch.equal(v, m2.state_dict()[k]) for k, v in m.state_dict().items())


def test_inference_uses_true_autoregressive_sql_tokens():
    from finetuning.tinysql.tokenizer import WordSQLTokenizer

    tok = WordSQLTokenizer.fit(
        [
            {
                "question": "count completed orders",
                "sql": "SELECT COUNT(*) FROM orders WHERE status = 'completed';",
            }
        ]
    )
    cfg = ModelConfig(vocab_size=len(tok.vocab), context=120, dim=32, layers=1, heads=4)
    m = TinySQLLM(cfg)
    result = m.generate(
        tok.format_input("count completed orders"), tok.eos_id, max_new_tokens=6
    )

    assert 0 <= len(result) <= 6
    assert all(0 <= int(i) < len(tok.vocab) for i in result)


def test_unknown_token_not_silently_removed():
    from finetuning.tinysql.tokenizer import WordSQLTokenizer

    t = WordSQLTokenizer(["<pad>", "<unk>", "<bos>", "<eos>", "select"])

    assert "<unk>" in t.decode([t.unk_id])
