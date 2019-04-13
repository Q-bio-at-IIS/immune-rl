fig2
----
Figure4の生成スクリプト。まずはこのレポジトリをcloneしてくる。
```
$ git clone git@github.com:Q-bio-at-IIS/immune-rl.git
$ cd immune-rl/fig4
```
次に実験データを取得する際に必要なライブラリをとってくる。Macbookの場合には
```
$ brew install sratoolkit
```
としてインストールできる。そして
```
$ sh prepare.sh
```
として実験データを取得することができる。最後に必要なpythonライブラリを導入したのちに
```
$ python main.py
```
としてFigure4を生成することができる。Figure4は`immune-rl/fig4`以下に`fig4.eps`として生成される。
