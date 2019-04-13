immune-rl
----
"Understanding adaptive immune system as reinforcement learning"のソースコード

## 環境
python3.6を使用している。依存ライブラリは
```
$ pip install -r requirements.txt
```
として導入できる。

## 構成
各ディレクトリfig2からfig5までが論文のfig2からfig5それぞれに対応しており、libディレクトリはそれらで共通の処理が書かれている。各Figureの生成の仕方はそれぞれのディレクトリ参照

## 注意
* 論文に乗せた図とは異なる random seed で計算している場合があるため、全く同じ図が生成されるとは限らない。
* シミュレーション結果は適当なディレクトリにキャッシュをしており、二度目以降同じシミュレーションをする場合には可能な限りキャッシュから結果を取り出すように設計されている。

## リンク
* https://arxiv.org/abs/1904.05581
