# consensus_demo
制御工学勉強会　第7回「コンセンサスな制御はいかが？」のデモプログラム

## コードの詳細
### class network()
ネットワークのグラフラプラシアンを生成する関数が格納されています．  
`topology.py`でトポロジーを確認できるようにするため，クラスとして定義を行っています．

``` python
class network():
    def __init__(self):
        self.L = np.empty

    def Laplacian(self) -> np.ndarray:
        # Set graph Laplacian matrix
        # If you want to change network, please edit the below matrix 'a'
        a = np.array([
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ])
        A = a + a.T
        D = np.diag(np.sum(A, axis=0))
        L = D - A

        return L
```

* `a`: 隣接行列の上三角部分
* `A`: 隣接行列（誰とつながっているのかを表現する行列）
* `D`: 次数行列（何人・何台とつながっているのかを表現する行列）
* `L`: グラフラプラシアン（ネットワークの接続全体について表現する行列）

### def Consensus(dimention: int, step: int)
合意制御のための制御入力の計算を行っている関数です．
ここでは離散時間系での合意アルゴリズムを記述しています．

```math
x[k+1] = x[k] - \epsilon \sum_{j \in \mathcal{N}_i} (x_i[k] - x_j[k])
```


```python
def Consensus(dimention: int, step: int) -> np.ndarray:
    G = network()
    L = G.Laplacian()
    n = L.shape[0] # Get the number of nodes from matrix shape
    x = np.zeros((n, dimention, step))
    x[:, :, 0] = np.random.uniform(size=(n, dimention)) # Generate the state vector with random function
    for k in range(1, step):
        x[:, :, k] = x[:, :, k - 1] - 0.05 * L @ x[:, :, k - 1] # Calculate control input
    return x
```

* `L`: `network`クラスで生成したグラフラプラシアン
* `n`: エージェント数
* `x`: 全エージェントの状態ベクトル

もし，コードが上手く動かない等ありましたら，Issue or Discordにてお知らせください．