Dockerコンテナの起動とコンテナへの入り方

コンテナ起動
```shell
docker compose up -d
```
コンテナへ入る
```shell
docker compose exec zemi-python-modules bash
```

仮想環境の起動
```bash
poetry shell
```

ライブラリの追加
```bash
poetry add pandas
```

必要なライブラリとバージョン
```
[tool.poetry.dependencies]
python = "^3.10"
pandas = "2.2.2"
janome = "0.4.2"
mojimoji = "^0.0.13"
scikit-learn = "^1.4.2"
torch = "^2.2.2"
transformers = "^4.39.3"
datasets = "^2.18.0"
scipy = "^1.13.0"
matplotlib = "^3.8.4"
japanize-matplotlib = "^1.1.3"
mca = "^1.0.3"
gensim = "^4.3.2"
tomotopy = "^0.12.7"
```