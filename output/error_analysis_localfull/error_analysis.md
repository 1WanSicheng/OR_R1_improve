# OR-R1 Error Diagnosis

This report diagnoses wrong `pass@1` samples from existing executed evaluation files. It is heuristic and intended to identify repairable error classes.

## Summary

| Dataset | Total | Correct | Wrong | Pass@1 | Top Wrong Categories |
|---|---:|---:|---:|---:|---|
| ComplexOR | 18 | 8 | 10 | 0.4444 | numeric/wrong_answer: 7, execution/runtime_error: 2, solver/non_optimal: 1 |
| ICMLTEST | 410 | 354 | 56 | 0.8634 | numeric/wrong_answer: 54, solver/non_optimal: 1, execution/runtime_error: 1 |
| IndustryOR | 100 | 28 | 72 | 0.2800 | numeric/wrong_answer: 41, solver/non_optimal: 20, execution/runtime_error: 11 |
| MAMO_ComplexLP | 211 | 92 | 119 | 0.4360 | numeric/wrong_answer: 73, solver/non_optimal: 42, execution/runtime_error: 4 |
| MAMO_EasyLP | 652 | 589 | 63 | 0.9034 | numeric/wrong_answer: 58, solver/non_optimal: 5 |
| NL4OPT | 230 | 216 | 14 | 0.9391 | numeric/wrong_answer: 14 |
| NLP4LP | 242 | 206 | 36 | 0.8512 | numeric/wrong_answer: 34, solver/non_optimal: 2 |
| OptiBench | 605 | 382 | 223 | 0.6314 | numeric/wrong_answer: 119, execution/runtime_error: 93, solver/non_optimal: 11 |

## Aggregate Wrong Categories

| Category | Count |
|---|---:|
| numeric/wrong_answer | 400 |
| execution/runtime_error | 111 |
| solver/non_optimal | 82 |

## Aggregate Diagnostic Tags

| Tag | Count |
|---|---:|
| possible_missing_upper/lower bounds | 194 |
| numeric_too_high | 189 |
| numeric_too_low | 171 |
| possible_missing_demand/minimum | 123 |
| runtime_error | 111 |
| possible_missing_capacity/budget/resource | 80 |
| possible_missing_inventory/time balance | 80 |
| missing_integrality_or_binary | 77 |
| infeasible | 74 |
| possible_missing_flow/supply-demand | 53 |
| near_miss | 40 |
| possible_missing_assignment/selection | 34 |
| unbounded | 8 |
| objective_direction_mismatch | 7 |
| very_few_constraints | 6 |

## Example Wrong Cases


### ComplexOR

**numeric/wrong_answer**
- pred=`9000.0`, gt=`8000.0`, state=`Execution Successful and Best Solution Found`, tags=['numeric_too_high']
  question: We have a set of flight legs (one-way non-stop flight) with a limited passenger capacity. According to market research, we defined a set of flight itineraries to sell as a package with a given price. For each package, we have an estimated d
- pred=`55.66666666666667`, gt=`66.0`, state=`Execution Successful and Best Solution Found`, tags=['missing_integrality_or_binary', 'possible_missing_capacity/budget/resource', 'numeric_too_low']
  question: Consider a problem where we have a set `P`. For each element `j` in `P`, we have a parameter `a[j]`, a parameter `c[j]`, and a parameter `u[j]`. We also have a global parameter `b`. We have a variable `X[j]` for each `j` in `P`. The goal is
**solver/non_optimal**
- pred=`No Best Solution`, gt=`235.0`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: This is a multi-commodity transportation problem. Given a set of origins `Origins`, a set of destinations `Destinations`, and a set of products `Products`. Each origin `i` has a certain supply of each product `p` `Supply_{i,p}` and each des
**execution/runtime_error**
- pred=`None`, gt=`10.0`, state=`Execution Failed: 2026-04-03 09:31:57 [INFO] checks license for COPT v8.0.3 20260113
2026-04-03 09:31:57 [WARN] no license files in current working folder: /mnt/workspace0/WWWWWWWWWWWW/OR-R1
2026-04-03 09:31:57 [WARN] no license files in binary folder: /home/wan/enter/bin
2026-04-03 09:31:57 [WARN] no license files in HOME folder: /home/wan/copt
2026-04-03 09:31:57 [INFO] empty environment variable: COPT_LICENSE_DIR
2026-04-03 09:31:57 [WARN] no license files in EV 'COPT_LICENSE_DIR': 

No license found. Starting COPT with size limitations for non-commercial use
Please apply for a license from www.shanshu.ai/copt

Cardinal Optimizer v8.0.3. Build date Jan 13 2026
Copyright Cardinal Operations 2026. All Rights Reserved

`, tags=['runtime_error']
  question: Consider a transportation problem with multiple products. Given a set of cities `Cities` and a set of links `Links` between the cities. Each city `i` has a certain supply of each product `p` `Supply_{i,p}` and a certain demand for each prod
- pred=`None`, gt=`0.0`, state=`Execution Failed: 2026-04-03 09:31:57 [INFO] checks license for COPT v8.0.3 20260113
2026-04-03 09:31:57 [WARN] no license files in current working folder: /mnt/workspace0/WWWWWWWWWWWW/OR-R1
2026-04-03 09:31:57 [WARN] no license files in binary folder: /home/wan/enter/bin
2026-04-03 09:31:57 [WARN] no license files in HOME folder: /home/wan/copt
2026-04-03 09:31:57 [INFO] empty environment variable: COPT_LICENSE_DIR
2026-04-03 09:31:57 [WARN] no license files in EV 'COPT_LICENSE_DIR': 

No license found. Starting COPT with size limitations for non-commercial use
Please apply for a license from www.shanshu.ai/copt

Cardinal Optimizer v8.0.3. Build date Jan 13 2026
Copyright Cardinal Operations 2026. All Rights Reserved

`, tags=['runtime_error']
  question: The Aircraft Landing Problem (ALP) is the problem of deciding a landing time on an appropriate runway for each aircraft in a given set of aircraft such that each aircraft lands within a predetermined time window; and separation criteria bet

### ICMLTEST

**numeric/wrong_answer**
- pred=`25.0`, gt=`260000.0`, state=`Execution Successful and Best Solution Found`, tags=['possible_missing_upper/lower bounds', 'numeric_too_low']
  question: A cosmetics company makes high-end skincare products whose main customers are wealthy women, both young girls and middle-aged women. In order to promote their product line, they decided to invest in short commercial spots on two types of pr
- pred=`68.0625`, gt=`78.0`, state=`Execution Successful and Best Solution Found`, tags=['missing_integrality_or_binary', 'numeric_too_low']
  question: A paint store mixes two brands of paint, Ruby and Sapphire, to create a new mixture of paint. A can of Ruby paint costs $12 and a can of Sapphire paint costs $15. A can of Ruby paint contains 2 units of dye, 4 units of thinner, and 5 units 
**solver/non_optimal**
- pred=`No Best Solution`, gt=`9649.57`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: A client asks his stockbroker to invest $100,000 for maximum annual income, subject to the following conditions: Spread the investment over no more than three different stocks. Put no more than 40 percent of the money into any one stock. Pu
**execution/runtime_error**
- pred=`None`, gt=`26200.0`, state=`Execution Failed: 2026-04-03 09:57:48 [INFO] checks license for COPT v8.0.3 20260113
2026-04-03 09:57:48 [WARN] no license files in current working folder: /mnt/workspace0/WWWWWWWWWWWW/OR-R1
2026-04-03 09:57:48 [WARN] no license files in binary folder: /home/wan/enter/bin
2026-04-03 09:57:48 [WARN] no license files in HOME folder: /home/wan/copt
2026-04-03 09:57:48 [INFO] empty environment variable: COPT_LICENSE_DIR
2026-04-03 09:57:48 [WARN] no license files in EV 'COPT_LICENSE_DIR': 

No license found. Starting COPT with size limitations for non-commercial use
Please apply for a license from www.shanshu.ai/copt

Cardinal Optimizer v8.0.3. Build date Jan 13 2026
Copyright Cardinal Operations 2026. All Rights Reserved

`, tags=['runtime_error']
  question: Brooks City has three consolidated high schools, each with a capacity of 1,200 students. The school board has partitioned the city into five busing districts—north, south, east, west, and central—each with different high school student popu

### IndustryOR

**numeric/wrong_answer**
- pred=`0.65`, gt=`-99999.0`, state=`Execution Successful and Best Solution Found`, tags=['numeric_too_high']
  question: A strategic bomber squadron has been ordered to destroy enemy military targets. It is known that there are four key areas, and destroying any one of them will achieve the objective. To complete this mission, the limits are set at $48000 \ma
- pred=`880.0`, gt=`734.0`, state=`Execution Successful and Best Solution Found`, tags=['possible_missing_capacity/budget/resource', 'possible_missing_demand/minimum', 'numeric_too_high']
  question: A factory produces three types of products: A, B, and C. Each unit of product A requires 1 hour of technical preparation, 10 hours of direct labor, and 3 kilograms of material. Each unit of product B requires 2 hours of technical preparatio
**solver/non_optimal**
- pred=`No Best Solution`, gt=`600.0`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: A company plans to transport goods between a city and suburb and needs to choose the most environmentally friendly mode of transportation. The company can choose from the following three options: motorcycles, small trucks, and large trucks.
- pred=`No Best Solution`, gt=`4685100.0`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: An Italian transportation company needs to transport some empty containers from its 6 warehouses (located in Verona, Perugia, Rome, Pescara, Taranto, and La Spezia) to major national ports (Genoa, Venice, Ancona, Naples, Bari). The inventor
**execution/runtime_error**
- pred=`None`, gt=`5004.0`, state=`Execution Failed: 2026-04-03 09:46:30 [INFO] checks license for COPT v8.0.3 20260113
2026-04-03 09:46:30 [WARN] no license files in current working folder: /mnt/workspace0/WWWWWWWWWWWW/OR-R1
2026-04-03 09:46:30 [WARN] no license files in binary folder: /home/wan/enter/bin
2026-04-03 09:46:30 [WARN] no license files in HOME folder: /home/wan/copt
2026-04-03 09:46:30 [INFO] empty environment variable: COPT_LICENSE_DIR
2026-04-03 09:46:30 [WARN] no license files in EV 'COPT_LICENSE_DIR': 

No license found. Starting COPT with size limitations for non-commercial use
Please apply for a license from www.shanshu.ai/copt

Cardinal Optimizer v8.0.3. Build date Jan 13 2026
Copyright Cardinal Operations 2026. All Rights Reserved

`, tags=['runtime_error']
  question: A project consists of the following 7 activities, with their durations (in days) as follows: $A(4), B(3), C(5), D(2), E(10), F(10), G(1)$. The following priorities are also given: $A \\rightarrow G, D ; E, G \\rightarrow F; D, F \\rightarro
- pred=`None`, gt=`770.0`, state=`Execution Failed: 2026-04-03 09:46:30 [INFO] checks license for COPT v8.0.3 20260113
2026-04-03 09:46:30 [WARN] no license files in current working folder: /mnt/workspace0/WWWWWWWWWWWW/OR-R1
2026-04-03 09:46:30 [WARN] no license files in binary folder: /home/wan/enter/bin
2026-04-03 09:46:30 [WARN] no license files in HOME folder: /home/wan/copt
2026-04-03 09:46:30 [INFO] empty environment variable: COPT_LICENSE_DIR
2026-04-03 09:46:30 [WARN] no license files in EV 'COPT_LICENSE_DIR': 

No license found. Starting COPT with size limitations for non-commercial use
Please apply for a license from www.shanshu.ai/copt

Cardinal Optimizer v8.0.3. Build date Jan 13 2026
Copyright Cardinal Operations 2026. All Rights Reserved

`, tags=['runtime_error']
  question: There are 8 villages in Tuanjie Township, with their respective coordinates and the number of elementary school students shown in Table 5-14.

Table 5-14
\begin{tabular}{c/c/c/c}
\hline Village Code & \multicolumn{2}{/c/}{Coordinate Positio

### MAMO_ComplexLP

**numeric/wrong_answer**
- pred=`63000.0`, gt=`63.0`, state=`Execution Successful and Best Solution Found`, tags=['missing_integrality_or_binary', 'numeric_too_high']
  question: Embark on a journey through a futuristic transportation network, connecting 8 bustling metropolises - from a cutting-edge logistics hub to a far-reaching distribution center. This network is not ordinary; it's a complex web of superhighways
- pred=`0.0`, gt=`9.0`, state=`Execution Successful and Best Solution Found`, tags=['numeric_too_low']
  question: There are two special nodes marked as S (likely the start) and T (likely the target or terminal). The other nodes are numbered from 2 to 7. Edges connect these nodes and each edge is labeled with a number indicating its weight. Node S is co
**execution/runtime_error**
- pred=`None`, gt=`43.0`, state=`Execution Failed: 2026-04-03 09:43:25 [INFO] checks license for COPT v8.0.3 20260113
2026-04-03 09:43:25 [WARN] no license files in current working folder: /mnt/workspace0/WWWWWWWWWWWW/OR-R1
2026-04-03 09:43:25 [WARN] no license files in binary folder: /home/wan/enter/bin
2026-04-03 09:43:25 [WARN] no license files in HOME folder: /home/wan/copt
2026-04-03 09:43:25 [INFO] empty environment variable: COPT_LICENSE_DIR
2026-04-03 09:43:25 [WARN] no license files in EV 'COPT_LICENSE_DIR': 

No license found. Starting COPT with size limitations for non-commercial use
Please apply for a license from www.shanshu.ai/copt

Cardinal Optimizer v8.0.3. Build date Jan 13 2026
Copyright Cardinal Operations 2026. All Rights Reserved

`, tags=['runtime_error']
  question: In the heart of a bustling metropolis, an expansive network of waterways and canals forms the lifeline for its residents, connecting 9 critical distribution centers that manage the flow of water from the city's reservoirs to its farthest su
- pred=`None`, gt=`62.0`, state=`Execution Failed: 2026-04-03 09:43:25 [INFO] checks license for COPT v8.0.3 20260113
2026-04-03 09:43:25 [WARN] no license files in current working folder: /mnt/workspace0/WWWWWWWWWWWW/OR-R1
2026-04-03 09:43:25 [WARN] no license files in binary folder: /home/wan/enter/bin
2026-04-03 09:43:25 [WARN] no license files in HOME folder: /home/wan/copt
2026-04-03 09:43:25 [INFO] empty environment variable: COPT_LICENSE_DIR
2026-04-03 09:43:25 [WARN] no license files in EV 'COPT_LICENSE_DIR': 

No license found. Starting COPT with size limitations for non-commercial use
Please apply for a license from www.shanshu.ai/copt

Cardinal Optimizer v8.0.3. Build date Jan 13 2026
Copyright Cardinal Operations 2026. All Rights Reserved

`, tags=['runtime_error']
  question: Welcome to the heart of an intricate transportation network, designed to efficiently distribute a critical resource across 8 bustling hubs of a futuristic city. Each hub, from a massive distribution center to the ultimate delivery point, is
**solver/non_optimal**
- pred=`No Best Solution`, gt=`165.0`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: Consider a delivery company that needs to deliver packages to five different cities, named E, F, G, H, and I. The delivery truck can start its route from any of these cities, but needs to visit each city exactly once and then return to the 
- pred=`No Best Solution`, gt=`142.0`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: Consider four cities: E, F, G, and H. A delivery driver is tasked with delivering packages to each of these cities. The driver can start their route from any one of these cities. However, the driver must ensure that they visit each city exa

### MAMO_EasyLP

**numeric/wrong_answer**
- pred=`63.0`, gt=`630000.0`, state=`Execution Successful and Best Solution Found`, tags=['numeric_too_low']
  question: In a human resources planning scenario, a company needs to allocate employees across three departments: $X1$, $X2$, and $X3$. These departments could represent different business units or functions within the organization. The total number 
- pred=`1500.0`, gt=`320000.0`, state=`Execution Successful and Best Solution Found`, tags=['possible_missing_capacity/budget/resource', 'possible_missing_upper/lower bounds', 'numeric_too_low']
  question: A financial advisor is managing a portfolio and plans to invest in three different assets: X, Y, and Z. The total investment across all three assets cannot exceed \$1000 due to budget constraints. Asset X requires a minimum investment of \$
**solver/non_optimal**
- pred=`No Best Solution`, gt=`1200.0`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: A farmer has four fields where he can grow Corn, Wheat, Soybean and Rice. Each field can be planted with one type of crop for the season. The planting restrictions are as follows: \n\n- The combined area of the fields planted with Corn and 
- pred=`No Best Solution`, gt=`5000.0`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: In a human resources scenario, a company is planning to allocate its training budget between two groups of employees: group $X$ and group $Y$. The total budget available for both groups combined cannot exceed $\$50,000$, and the company wan

### NL4OPT

**numeric/wrong_answer**
- pred=`236.5`, gt=`224.0`, state=`Execution Successful and Best Solution Found`, tags=['possible_missing_demand/minimum', 'possible_missing_inventory/time balance', 'near_miss']
  question: A chair produced by Elm Furniture yields a profit of $43, while every dresser yields a $52 profit. Each week, 17 gallons of stain and 11 lengths of oak wood are available. Each chair requires 1.4 gallons of stain and 2 lengths of oak wood, 
- pred=`1400.0`, gt=`700.0`, state=`Execution Successful and Best Solution Found`, tags=['possible_missing_capacity/budget/resource', 'possible_missing_demand/minimum', 'possible_missing_upper/lower bounds', 'numeric_too_high']
  question: A cleaning company uses a cleansing chemical and odor-removing chemical to clean a house. Each unit of the cleansing chemical takes 4 minutes to be effective while each unit of the odor-removing chemical takes 6 minutes to be effective. The

### NLP4LP

**numeric/wrong_answer**
- pred=`0.0`, gt=`26.0`, state=`Execution Successful and Best Solution Found`, tags=['possible_missing_upper/lower bounds', 'numeric_too_low']
  question: A bakery makes fiber supplemented brownies and lemon squares. Each brownie requires 5 units of chocolate mix and 4 units of fiber. Each lemon square requires 7 units of lemon mix and 6 units of fiber. Lemon squares sell much faster and thus
- pred=`1400.0`, gt=`700.0`, state=`Execution Successful and Best Solution Found`, tags=['missing_integrality_or_binary', 'possible_missing_capacity/budget/resource', 'possible_missing_demand/minimum', 'possible_missing_upper/lower bounds', 'numeric_too_high']
  question: A cleaning company uses a cleansing chemical and odor-removing chemical to clean a house. Each unit of the cleansing chemical takes 4 units to be effective while each unit of the odor-removing chemical takes 6 minutes to be effective. The c
**solver/non_optimal**
- pred=`No Best Solution`, gt=`0.0`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: Both chorine and water softener need to be added to a pool. One unit of chlorine takes 1 minute to be effective while one unit of water softener takes 2 minutes to be effective. Because too much chlorine can burn your eyes, there has to at 
- pred=`No Best Solution`, gt=`33.5`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: A chicken farmer has sold his chicken and they need to be transported either by bus or by car. A bus can take 100 chicken and takes 2 hours per trip. A car can take 40 chicken and takes 1.5 hours per trip. There can be at most 10 bus trips 

### OptiBench

**numeric/wrong_answer**
- pred=`642675.0`, gt=`85500.0`, state=`Execution Successful and Best Solution Found`, tags=['numeric_too_high']
  question: A fashion company sells regular handbags and premium handbags made of higher quality material. They can sell regular handbags at a profit of $30 each and premium handbags at a profit of $180 each. The total monthly cost of manufacturing is 
- pred=`0.0`, gt=`18.0`, state=`Execution Successful and Best Solution Found`, tags=['possible_missing_upper/lower bounds', 'numeric_too_low']
  question: A smoothie shop has a promotion for their two smoothies; an acai berry smoothie and a banana chocolate smoothie. It takes 7 units of acai berries and 3 units of water to make the acai berry smoothie. It takes 6 units of banana chocolate and
**execution/runtime_error**
- pred=`None`, gt=`40.0`, state=`Execution Failed: 2026-04-03 09:53:59 [INFO] checks license for COPT v8.0.3 20260113
2026-04-03 09:53:59 [WARN] no license files in current working folder: /mnt/workspace0/WWWWWWWWWWWW/OR-R1
2026-04-03 09:53:59 [WARN] no license files in binary folder: /home/wan/enter/bin
2026-04-03 09:53:59 [WARN] no license files in HOME folder: /home/wan/copt
2026-04-03 09:53:59 [INFO] empty environment variable: COPT_LICENSE_DIR
2026-04-03 09:53:59 [WARN] no license files in EV 'COPT_LICENSE_DIR': 

No license found. Starting COPT with size limitations for non-commercial use
Please apply for a license from www.shanshu.ai/copt

Cardinal Optimizer v8.0.3. Build date Jan 13 2026
Copyright Cardinal Operations 2026. All Rights Reserved

`, tags=['runtime_error']
  question: Of all rectangles of area 100, which has the smallest perimeter?
- pred=`None`, gt=`20.0`, state=`Execution Failed: 2026-04-03 09:53:59 [INFO] checks license for COPT v8.0.3 20260113
2026-04-03 09:53:59 [WARN] no license files in current working folder: /mnt/workspace0/WWWWWWWWWWWW/OR-R1
2026-04-03 09:53:59 [WARN] no license files in binary folder: /home/wan/enter/bin
2026-04-03 09:53:59 [WARN] no license files in HOME folder: /home/wan/copt
2026-04-03 09:53:59 [INFO] empty environment variable: COPT_LICENSE_DIR
2026-04-03 09:53:59 [WARN] no license files in EV 'COPT_LICENSE_DIR': 

No license found. Starting COPT with size limitations for non-commercial use
Please apply for a license from www.shanshu.ai/copt

Cardinal Optimizer v8.0.3. Build date Jan 13 2026
Copyright Cardinal Operations 2026. All Rights Reserved

No license found. The size is limited to 2000 variables and 2000 constraints
Please apply for a license from www.shanshu.ai/copt

Model fingerprint: cd3f8529

Warning: MINLP problem is not supported
`, tags=['runtime_error']
  question: A bakery produces five types of cakes: C1, C2, C3, C4, and C5. They need to determine the quantities of each cake to produce. For C2, the revenue per unit is $40, the production time per unit is 2 hours, and the ingredient cost per unit is 
**solver/non_optimal**
- pred=`No Best Solution`, gt=`2.69`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: A manufacturing company produces three types of electronic devices: smartphones, tablets, and laptops. The company has four different production lines, each with varying efficiency and capacity. On production line 1, each worker produces 10
- pred=`No Best Solution`, gt=`15.0`, state=`Execution Successful but No Best Solution Found`, tags=['infeasible']
  question: A high rise building is buying two types of furnaces, a new model and an old model. A new model furnace can heat 10 apartments and consumes 200 kWh per day. An old model can heat 15 apartments and consumes 250 kWh per day. Since the old mod
