### 問題文

AtCoder社はトップページにスポンサー企業 n 社のWeb広告を設置することにした。 広告を設置するスペースは 10000\times 10000
の正方形をしており、各企業毎の広告スペースは軸に平行な長方形で正の面積を持ち、頂点の座標は整数値でなければならない。
異なる長方形は辺を接してもよいが、重なってはならない。つまり、共通部分が正の面積を持ってはならない。 どの広告にも属さない空きスペースが残っても構わない。

高橋社長が各企業に希望する位置と面積を尋ねた結果、企業 i は点 (x_i+0.5, y_i+0.5) を含む面積 r_i
の広告スペースを希望していることが分かった。 企業 i の満足度 p_i は以下のように定まる。

  * 企業 i の広告スペースが点 (x_i+0.5, y_i+0.5) を含まない場合、p_i = 0
  * 企業 i の広告スペースが点 (x_i+0.5, y_i+0.5) を含む場合、面積を s_i として p_i = 1 - (1 - \min(r_i,s_i) / \max(r_i, s_i))^2

満足度の総和が出来るだけ大きくなるように広告スペースの配置を決定せよ。 10^9 \times \sum_{i=0}^{n-1} p_i / n
を最も近い整数に丸めた得点が得られる。

![](https://img.atcoder.jp/ahc001/edaf52926517566f9dd8018d402400ec.png)

* * *

### 入力

入力は以下の形式で標準入力から与えられる。



    n
    x_0 y_0 r_0
    \vdots
    x_{n-1} y_{n-1} r_{n-1}


  * 50\leq n\leq 200
  * x_i, y_i は整数値であり、0\leq x_i\leq 9999、0\leq y_i\leq 9999 を満たす。全ての i\neq j に対し、(x_i,y_i)\neq (x_j,y_j) が成り立つ。
  * r_i は 1 以上の整数値であり、\sum_{i=0}^{n-1} r_i=10000\times 10000 を満たす。

### 出力

企業 i の広告スペースを表す長方形の対角となる2頂点の座標を (a_i, b_i), (c_i, d_i) (0\leq a_i<c_i\leq
10000, 0\leq b_i<d_i\leq 10000) としたとき、以下の形式で標準出力に出力せよ。



    a_0 b_0 c_0 d_0
    \vdots
    a_{n-1} b_{n-1} c_{n-1} d_{n-1}


### 入力生成方法

0 以上 1 未満の倍精度浮動小数点数を一様ランダムに生成する関数を rand() で表す。

#### n の生成

企業数 n の値は 50 × 4^{rand()} を最も近い整数値に丸めることで生成される。

#### x_i, y_i の生成

\\{(x, y) \mid x\in \\{0,1,\ldots,9999\\}, y\in\\{0,1,\ldots,9999\\}\\}
の中から異なる座標をランダムに n 個サンプルすることで生成される。

#### r_i の生成

\\{1,2,\ldots,99999999\\}の中から異なる値をランダムに n-1 個サンプルし、それをソートした列を
q_1,\ldots,q_{n-1} とする。 q_0=0、q_n=100000000 として、r_i=q_{i+1}-q_i とする。

### テストケース数

  * 暫定テスト 50個
  * システムテスト 1000個、 コンテスト終了後に[seeds.txt](https://img.atcoder.jp/ahc001/seeds.zip)(md5=8fc1ce3f4beabac6abc1bdb4206d7f7e)を公開

各テストケースの得点の合計が提出の得点となる。
暫定テストでは、一つ以上のテストケースで不正な出力や制限時間超過をした場合、提出全体の判定がWAやTLEとなる。
システムテストでは、不正な出力や制限時間超過をした場合、そのテストケースのみ0点となる。

### ツール

入力ジェネレータとビジュアライザは
[ここ](https://img.atcoder.jp/ahc001/ded8fd3366b4ff0b0d7d053f553cdb84.zip)
からダウンロード出来ます。 使用するには、[Rust言語](https://www.rust-lang.org/ja)のコンパイル環境をご用意下さい。
サポート対象外ですが、kenkoooo さんによりWeb上からも利用できるようになりました。
<https://kenkoooo.github.io/ahc001-gen-vis-wasm/>

* * *

### 入力例 1



    50
    1909 360 6468907
    5810 7091 4661329
    5407 422 2010076
    5767 3140 681477
    6659 3234 920591
    4206 1620 2487369
    7853 9492 440133
    7875 432 586159
    9048 5059 1805425
    7292 9070 509242
    7633 2496 1558444
    421 4835 1808752
    7164 4109 35081
    5356 2271 78438
    5261 577 971398
    3546 5225 1871979
    4667 3386 28796
    5596 7896 3310195
    2518 9813 1739130
    9002 3913 334620
    8574 8947 1107057
    3118 1773 669849
    7140 4388 2098247
    8544 8196 1742491
    8577 4337 4435283
    3155 9168 976005
    7823 4404 945830
    9451 110 569854
    7031 1389 787729
    1841 2337 942236
    76 8364 710110
    3543 3931 3840994
    3927 8828 2920828
    5671 3305 1526349
    5542 4587 6285390
    4030 7732 3962404
    8575 8200 3662259
    1139 3739 254000
    50 7415 647735
    934 4056 1800657
    8801 7178 1218595
    4499 6207 660560
    3096 3375 2695827
    5252 3281 1046149
    2247 1446 7148429
    3347 8501 7546190
    5791 8600 3909497
    8033 8992 3365971
    2297 9254 23830
    4312 6176 192104


### 出力例 1



    0 0 4473 1446
    4634 5915 6987 7896
    4473 0 7875 577
    5260 2633 6274 3305
    6274 2722 7172 3747
    3174 1446 5238 2651
    7522 9161 8185 9824
    7875 0 8454 1012
    8376 4388 9720 5731
    6890 8668 7522 9473
    7172 1815 8315 3178
    0 4072 1185 5598
    7071 4016 7258 4203
    5238 2124 5504 2418
    4565 577 5958 1274
    2862 4541 4230 5909
    4582 3302 4752 3471
    3348 7896 8049 8600
    1351 9255 3685 10000
    8713 3624 9291 4202
    8185 8373 9149 9521
    2480 1446 3174 2411
    6105 4203 7823 5424
    8049 5731 8575 8373
    7823 3178 8713 4388
    2410 8600 3900 9255
    7823 4388 8376 5731
    8969 0 9933 591
    6581 940 7481 1815
    1356 1852 2327 2822
    0 7805 635 8923
    1286 3376 4582 4541
    3900 8600 5791 10000
    4752 3305 6274 4203
    4582 4203 6105 5915
    2287 6208 4634 7896
    8575 7179 10000 8373
    849 3449 1286 4030
    0 6812 652 7805
    0 4030 1286 4072
    8575 6182 9797 7179
    4313 5915 4634 6208
    2327 2651 4582 3376
    4752 2651 5260 3305
    0 1446 2480 1852
    635 7896 3348 8600
    5791 8600 6890 10000
    7522 8600 8185 9161
    2188 9146 2406 9255
    3671 5909 4313 6208


### Problem Statement

AtCoder has decided to place web advertisements of n companies on the top
page. The space for placing advertisements is a square of size 10000 x 10000.
The space for each company must be an axis-parallel rectangle with positive
area, and the coordinates of the vertices must be integer values. Different
rectangles may touch on their sides, but they must not overlap. In other
words, the common area must not have positive area. It is allowed to leave
some free space that does not belong to any ad.

President Takahashi asked each company for their desired location and area.
Company i wants an ad space with area r_i including point (x_i+0.5, y_i+0.5).
The satisfaction level p_i of company i is determined as follows.

  * If the ad space for company i does not contain the point (x_i+0.5, y_i+0.5), then p_i = 0.
  * If the ad space for company i contains the point (x_i+0.5, y_i+0.5) and the area is s_i, then p_i = 1 - (1 - \min(r_i,s_i) / \max(r_i, s_i))^2.

Your task is to determine the placement of the ads so that the sum of the
satisfaction levels is maximized. You will get a score of 10^9 \times
\sum_{i=0}^{n-1} p_i / n rounded to the nearest integer.

![](https://img.atcoder.jp/ahc001/edaf52926517566f9dd8018d402400ec.png)

* * *

### Input

Input is given from Standard Input in the following format:



    n
    x_0 y_0 r_0
    \vdots
    x_{n-1} y_{n-1} r_{n-1}


  * 50\leq n\leq 200
  * x_i and y_i are integers satisfying 0\leq x_i\leq 9999 and 0\leq y_i\leq 9999. For any i\neq j, (x_i,y_i)\neq (x_j,y_j) holds.
  * r_i is an integer at least one and satisfies \sum_{i=0}^{n-1} r_i=10000\times 10000.

### Output

Let (a_i, b_i) and (c_i, d_i) (0\leq a_i<c_i\leq 10000, 0\leq b_i<d_i\leq
10000) be the coordinates of the two diagonal vertices of the rectangle
representing the ad space for company i. Output to standard output in the
following format.



    a_0 b_0 c_0 d_0
    \vdots
    a_{n-1} b_{n-1} c_{n-1} d_{n-1}


### Input Generation

Let rand() be a function that generates a uniformly random double-precision
floating point number at least zero and less than one.

#### Generation of n

The number of companies n is generated by rounding 50 × 4^{rand()} to the
nearest integer value.

#### Generation of x_i and y_i

The list of desired locations (x_1,y_i),\ldots,(x_n,y_n) is generated by
randomly sampling n distinct coordinates from \\{(x, y) \mid x\in
\\{0,1,\ldots,9999\\}, y\in\\{0,1,\ldots,9999\\}\\}.

#### Generation of r_i

Let q_1,\ldots,q_{n-1} be a sorted list of n-1 distinct integers randomly
sampled from \\{1,2,\ldots,99999999\\}. Let q_0=0 and q_n=100000000. Then
r_i=q_{i+1}-q_i.

### Number of test cases

  * Provisional test: 50
  * System test: 1000. We will publish [seeds.txt](https://img.atcoder.jp/ahc001/seeds.zip) (md5=8fc1ce3f4beabac6abc1bdb4206d7f7e) after the contest is over.

The score of a submission is the total scores for each test case. In the
provisional test, if your submission produces illegal output or exceeds the
time limit for some test cases, the submission itself will be judged as WA or
TLE, and the score of the submission will be zero. In the system test, if your
submission produces illegal output or exceeds the time limit for some test
cases, only the score for those test cases will be zero.

### Tools

You can download an input generator and visualizer
[here](https://img.atcoder.jp/ahc001/ded8fd3366b4ff0b0d7d053f553cdb84.zip). To
use them, you need a compilation environment of [Rust
language](https://www.rust-lang.org/ja). Thanks to kenkoooo, You can now use
the tools on the web, although they are not officially supported.
<https://kenkoooo.github.io/ahc001-gen-vis-wasm/>

* * *

### Sample Input 1



    50
    1909 360 6468907
    5810 7091 4661329
    5407 422 2010076
    5767 3140 681477
    6659 3234 920591
    4206 1620 2487369
    7853 9492 440133
    7875 432 586159
    9048 5059 1805425
    7292 9070 509242
    7633 2496 1558444
    421 4835 1808752
    7164 4109 35081
    5356 2271 78438
    5261 577 971398
    3546 5225 1871979
    4667 3386 28796
    5596 7896 3310195
    2518 9813 1739130
    9002 3913 334620
    8574 8947 1107057
    3118 1773 669849
    7140 4388 2098247
    8544 8196 1742491
    8577 4337 4435283
    3155 9168 976005
    7823 4404 945830
    9451 110 569854
    7031 1389 787729
    1841 2337 942236
    76 8364 710110
    3543 3931 3840994
    3927 8828 2920828
    5671 3305 1526349
    5542 4587 6285390
    4030 7732 3962404
    8575 8200 3662259
    1139 3739 254000
    50 7415 647735
    934 4056 1800657
    8801 7178 1218595
    4499 6207 660560
    3096 3375 2695827
    5252 3281 1046149
    2247 1446 7148429
    3347 8501 7546190
    5791 8600 3909497
    8033 8992 3365971
    2297 9254 23830
    4312 6176 192104


### Sample Output 1



    0 0 4473 1446
    4634 5915 6987 7896
    4473 0 7875 577
    5260 2633 6274 3305
    6274 2722 7172 3747
    3174 1446 5238 2651
    7522 9161 8185 9824
    7875 0 8454 1012
    8376 4388 9720 5731
    6890 8668 7522 9473
    7172 1815 8315 3178
    0 4072 1185 5598
    7071 4016 7258 4203
    5238 2124 5504 2418
    4565 577 5958 1274
    2862 4541 4230 5909
    4582 3302 4752 3471
    3348 7896 8049 8600
    1351 9255 3685 10000
    8713 3624 9291 4202
    8185 8373 9149 9521
    2480 1446 3174 2411
    6105 4203 7823 5424
    8049 5731 8575 8373
    7823 3178 8713 4388
    2410 8600 3900 9255
    7823 4388 8376 5731
    8969 0 9933 591
    6581 940 7481 1815
    1356 1852 2327 2822
    0 7805 635 8923
    1286 3376 4582 4541
    3900 8600 5791 10000
    4752 3305 6274 4203
    4582 4203 6105 5915
    2287 6208 4634 7896
    8575 7179 10000 8373
    849 3449 1286 4030
    0 6812 652 7805
    0 4030 1286 4072
    8575 6182 9797 7179
    4313 5915 4634 6208
    2327 2651 4582 3376
    4752 2651 5260 3305
    0 1446 2480 1852
    635 7896 3348 8600
    5791 8600 6890 10000
    7522 8600 8185 9161
    2188 9146 2406 9255
    3671 5909 4313 6208
