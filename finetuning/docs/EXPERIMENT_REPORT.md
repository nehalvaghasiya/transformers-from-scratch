# Full evaluation and experimental integrity report

## Real measured execution accuracy

| Split | n | Base | LoRA | Gain pp | Word TF-IDF + slots | Character TF-IDF |
|---|---:|---:|---:|---:|---:|---:|
| val | 320 | 69.38% | 75.94% | +6.56 | 78.44% | 65.31% |
| test | 320 | 73.44% | 88.75% | +15.31 | 64.06% | 52.81% |
| challenge | 160 | 22.50% | 22.50% | +0.00 | 78.12% | 69.38% |

The scores use raw autoregressive decoding and matched execution results on two populated synthetic databases. The challenge split has no improvement despite strong test-set gains. Challenge SQL validity
falls from **41.25% for the base to 25.00% for LoRA**.

The canonical historical release is `adapters/aug_r8_fast`. Rank 4 has lower recorded
validation CE than rank 8; the complete cross-configuration release selection rationale is
unavailable. Validation-loss checkpoint selection applies within each training run.

## Per-SQL-family breakdown

### val

| Family | n | Frozen base | LoRA | Gain pp |
|---|---:|---:|---:|---:|
| order_count | 8 | 100.00% | 100.00% | +0.00 |
| order_lookup | 82 | 76.83% | 86.59% | +9.76 |
| order_sum | 8 | 100.00% | 100.00% | +0.00 |
| product_avg | 16 | 100.00% | 100.00% | +0.00 |
| product_lookup | 77 | 54.55% | 66.23% | +11.69 |
| state_customers | 24 | 83.33% | 100.00% | +16.67 |
| state_order_count | 64 | 56.25% | 56.25% | +0.00 |
| top_customers | 41 | 70.73% | 70.73% | +0.00 |

### test

| Family | n | Frozen base | LoRA | Gain pp |
|---|---:|---:|---:|---:|
| order_count | 8 | 50.00% | 50.00% | +0.00 |
| order_lookup | 84 | 83.33% | 100.00% | +16.67 |
| order_sum | 8 | 100.00% | 100.00% | +0.00 |
| product_avg | 16 | 100.00% | 100.00% | +0.00 |
| product_lookup | 66 | 60.61% | 68.18% | +7.58 |
| state_customers | 24 | 66.67% | 100.00% | +33.33 |
| state_order_count | 69 | 52.17% | 84.06% | +31.88 |
| top_customers | 45 | 100.00% | 100.00% | +0.00 |

### challenge

| Family | n | Frozen base | LoRA | Gain pp |
|---|---:|---:|---:|---:|
| order_count | 4 | 0.00% | 0.00% | +0.00 |
| order_lookup | 53 | 0.00% | 0.00% | +0.00 |
| order_sum | 4 | 100.00% | 100.00% | +0.00 |
| product_avg | 8 | 100.00% | 100.00% | +0.00 |
| product_lookup | 34 | 0.00% | 0.00% | +0.00 |
| state_customers | 10 | 0.00% | 0.00% | +0.00 |
| state_order_count | 23 | 0.00% | 0.00% | +0.00 |
| top_customers | 24 | 100.00% | 100.00% | +0.00 |

## Select prediction failure examples

Sample outputs below are **real generated SQL**, not hand-crafted examples.

### test
**Question:** find products classified beauty whose price is at most 300
- Gold: `SELECT name FROM products WHERE category = 'beauty' AND price <= 300 ORDER BY price ASC, name ASC;`
- LoRA: `select name from products where category = 'beauty' and price <= 300 order by price asc ;`

**Question:** i need number of pending orders for nc customers
- Gold: `SELECT COUNT(*) FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.state = 'nc' AND o.status = 'pending';`
- LoRA: `select count ( * ) from orders o join customers c on o . customer_id = c . id where c . state = ' and o . status = 'pending' ;`

**Question:** find products classified electronics whose price is at most 75
- Gold: `SELECT name FROM products WHERE category = 'electronics' AND price <= 75 ORDER BY price ASC, name ASC;`
- LoRA: `select name from products where category = 'electronics' and price <= 75 order by price asc ;`

**Question:** find products classified beauty whose price is at most 500
- Gold: `SELECT name FROM products WHERE category = 'beauty' AND price <= 500 ORDER BY price ASC, name ASC;`
- LoRA: `select name from products where category = 'beauty' and price <= 500 order by price asc ;`

### challenge
**Question:** provide a count for status pending in orders
- Gold: `SELECT COUNT(*) FROM orders WHERE status = 'pending';`
- LoRA: `select count ( * ) from orders o join customers c on o . customer_id = c . id where c . state = 'ny' and o . status = 'pending' ;`

**Question:** i need 5 ids for orders at least 500 in refunded sorted largest first
- Gold: `SELECT id FROM orders WHERE status = 'refunded' AND total >= 500 ORDER BY total DESC, id ASC LIMIT 5;`
- LoRA: `select sum ( total ) from orders where status = 'refunded' refunded ' and total >= 500 order by total >= 500 order by total ) from orders o ;`

**Question:** i am interested in names of people from wa sort by name
- Gold: `SELECT name FROM customers WHERE state = 'wa' ORDER BY name ASC;`
- LoRA: `select name from customers where state = 'wa' order by price asc , name asc ;`

**Question:** i am interested in names of people from oh sort by name
- Gold: `SELECT name FROM customers WHERE state = 'oh' ORDER BY name ASC;`
- LoRA: `select name from customers where state = 'oh' order by price asc , name asc ;`

## Caveats

- Synthetic fixed-schema corpus, substantial repeated SQL skeletons, and only a small tokenizer (unknown words become `<unk>`).
- Original model development previously evaluated the test/challenge sets; these are historical, not newly blind.
- Model configuration chosen by validation; score uncertainty estimates are paired bootstrap over observed rows, not full training-seed uncertainty.
- The base was pretrained from scratch in a previous project; here only low-rank adapters were optimized.
- The conventional baselines may be highly competitive, especially character TF-IDF and slot copying on novel wording.
- Stronger grammar-based semantic parsers and larger pretrained text-to-SQL models have not been benchmarked in this run.
