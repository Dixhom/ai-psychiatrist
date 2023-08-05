# はじめに

もし AI が医師になったら・・・そんな未来を考えたことがありますか？医療の参入障壁は下がり、診療費も格安となり、誰もがいつでもどこでもスマホに症状を言うだけで治療が受けられるでしょう。本稿はそんな未来を夢見た実験です。

# 作った理由

精神科は問診中心で診断を下すため、テキストデータのみで疾患の判定がしやすいのではないかと考えたため（実際は話し方や表情、身なりも見ているらしい）。

# 注意

**文中の症例は医学的な訓練を受けていない私が限定された知識を元に作ったもので、実際の疾患を正しく反映しているとは限りません。**

# データ

外部から取得、整形したデータを列"q"列（質問、プロンプト）、"a"列（回答、コンプリーション）のニ列を持つ csv ファイルに保存しています（非公開）。

# Openai キーの設定

最初に Openai キーを環境変数に設定します。これは生の値をハードコードしたくないため、ローカルファイルから読み込む形を取っています。適宜読み替えて下さい。

```python
def get_openai_apikey():
    """get openai api key"""
    # パスは適宜読み替えて下さい
    path = r'F:\secrets\openai_apikey.txt'
    if os.path.exists(path): # localhost
        with open(path, 'r') as f:
            return f.read().replace('\n', '')
    else:
        raise Exception('no key file!')

import os
import openai

os.environ["OPENAI_API_KEY"] = get_openai_apikey()
```

# データ読み込み

```python
# パスは適宜読み替えて下さい
df = pd.read_csv('./data/pqa.csv', encoding='utf-8-sig')
```

# データの準備

openai の API に渡すデータは長すぎるとエラーを出すので切り詰めます。

```python
def n_token(s):
    # 文字数からトークン数を推定
    # 数式の根拠は失念
    return 1.35049 * len(s) + 2.92158

df['ntokenq'] = df.q.apply(n_token)
df['ntokena'] = df.a.apply(n_token)

# question ,answerともにtoken 2000未満のレコード
dff = df[(df.ntokenq < 2000) & (df.ntokena < 2000)]
```

重複を削除します。

```python
# 重複削除
dff = dff.drop_duplicates(subset='q').drop_duplicates(subset='a')
```

データを jsonl 形式で準備します。

```python
# データの準備（jsonl）
# Data Formattingは以下参照
# https://platform.openai.com/docs/guides/fine-tuning/data-formatting
df_jsonl = pd.DataFrame({'prompt': dff.q + '\n\n###\n\n', 'completion': ' ' + dff.a + '###'})
df_jsonl.to_json('./data/qas.jsonl', orient='records', force_ascii=False, lines=True)
```

openai にデータをアップロードします。

```
!openai tools fine_tunes.prepare_data -f ./data/qas.jsonl -q
```

```
Analyzing...

- Your file contains 4509 prompt-completion pairs
- All prompts end with suffix `\n\n###\n\n`
- All completions end with suffix `###`

No remediations found.

You can use your file for fine-tuning:
> openai api fine_tunes.create -t "./data/qas.jsonl"

After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string `\n\n###\n\n` for the model to start generating completions, rather than continuing with the prompt. Make sure to include `stop=["###"]` so that the generated texts ends at the expected place.
Once your model starts training, it'll approximately take 1.07 hours to train a `curie` model, and less for `ada` and `babbage`. Queue will approximately take half an hour per job ahead of you.
```

訓練開始します。

```
!openai api fine_tunes.create -t "./data/qas.jsonl"
```

```
[2023-05-18 07:56:43] Created fine-tune: ft-XXXXXXXXXXXXXXXXXXXXXXXXXX
[2023-05-18 08:01:31] Fine-tune costs $8.48
[2023-05-18 08:01:32] Fine-tune enqueued. Queue number: 0
[2023-05-18 08:01:32] Fine-tune started
[2023-05-18 08:09:56] Completed epoch 1/4
[2023-05-18 08:25:02] Completed epoch 3/4
[2023-05-18 08:33:01] Uploaded model: ada:ft-personal-2023-XXXXXXXXXXXXX
[2023-05-18 08:33:02] Uploaded result file: file-XXXXXXXXXXXXXXXXXXX
[2023-05-18 08:33:02] Fine-tune succeeded

Job complete! Status: succeeded 🎉
Try out your fine-tuned model:

openai api completions.create -m ada:ft-personal-2023-XXXXXXXXXXXX -p <YOUR_PROMPT>
```

無事訓練できました。コンプリーション生成のためのメソッドを作っておきます。

```python
def get_answer(q):
    ft_model = 'ada:ft-personal-2023-05-17-23-33-01'
    res = openai.Completion.create(
        model=ft_model,
        prompt=q + '\n\n###\n\n',
        max_tokens=1000,
        temperature=0.8,
        stop='###')
    a = res['choices'][0]['text']
    return a
```

# テスト

次に、各症例のプロンプトを入力して妥当なコンプリーションが返ってくるかを確認します。繰り返しになりますが、**文中の症例は医学的な訓練を受けていない私が限定された知識を元に作ったもので、実際の疾患を正しく反映しているとは限りません。**

## 統合失調症

```python
q = '''28歳の男性です。高校までは明るく友人も多かったのですが、大学に進学した後、どことなく不安な感じが出現して休学しました。説明が難しいのですが、世界が不気味に変容して襲いかかってくる感じがしました。

その後、テレビを点けると出演者が突然ぎこちない話し方をします。おそらく、僕がテレビをONするとそれがスタジオに伝わる仕掛けになっているのだと思います。出演者に見られて笑われている気もします。

外出するとスパイのような人たちがうろうろしていて、自分を殺そうとしているのではないかと思いました。道を歩いていると通行人がすれ違いざまに「ばか」「消えろ」と悪口を言ってきます。夜部屋の明かりを点けると、それに合わせてお隣さんも照明を点けます。

最近気づいたことですが部屋に無数の盗聴器や監視カメラが仕掛けられています。常に見られている感覚がします。業者を呼んで取り除いてもらおうか考えています。

私は病気でしょうか。'''

a = get_answer(q)
print(a)
```

```
 解離性障害か、統合失調症でしょう。精神科を受診してください。
```

可能性を 2 つ述べていますが、統合失調症を言い当てています。

## 双極性障害

```python
q = '''30代男性です。前から調子がおかしく、「自分は素晴らしい人物だ」「皆が自分を称賛すべきだ」という考えに囚われることがあります。

また、浪費癖が時々出てしまい、ブランド物を衝動的に買い漁って後で後悔することがあります。お金を使いすぎて数百万借金しました。
その借金も返せなくなり、最終的に自己破産しました。

途端に気分が優れなくなり、一日中寝ているようになりました。食欲が減り、体重が15kg減りました。思考もまとまらず、意欲がどうしても湧きません。

私は病気でしょうか？
'''

a = get_answer(q)
print(a)
```

```
 [quote]前から調子がおかしく、「自分は素晴らしい人物だ」「皆が自分を称賛すべきだ」という考えに囚われることがあります。[/quote]

      あります。統合失調症でしょう。「昔は素晴らしい人物が自分だった」と考えてしまうそこで、回復に向かうのは非常に難しいところです。

[quote]途端に気分が優れなくなり、一日中寝ているようになりました。食欲が減り、体重が15kg減りました。思考もまとまらず、意欲がどうしても湧きません。[/quote]

      このような状態では、慎重にこのようにお考えになることは稀で、統合失調症の良い時期には少なからず誤りが出てきます。結果として今の状態は、それが悪いと悪いというよりも、悪いと悪いという方がよくあることです。

[quote]私は病気でしょうか？[/quote]

      そうです、病気です。精神科を受診してください。そうすれば今の苦しさは解消に向かうと思います。
```

統合失調症だと誤診しています。文も、文法的な乱れが多いです。

## うつ病

```python
q = '''40代女性です。悲しい気分が止まりません。自分は価値のない人間だと思います。
5階の窓から外を見下ろして「死にたい」と感じます。気分の落ち込みや、不安感、対人恐怖に苦しんでいます。夜もよく眠れません。
私は何病でしょうか？
'''

a = get_answer(q)
print(a)
```

```
 病気かもしれないと思います。普通は夜無く、経過が曖昧ですが、病気だった場合、夜に出ることがありえます。その場合は死にたいと感じることがあります。精神科を受診してください。
```

病名の指摘がありません。

## 相貌失認

```python
q = '''20代女性です。人の顔が全く覚えられません。どんなに人の顔の特徴を覚えようとしてもだめです。
顔そのものはうまく覚えられないため、声や髪型、ほくろや眼鏡などで見分けています。
先日も、よく知っている人に声をかけられたのに誰だか分からずに失礼な対応をしてしまいました。
ドラマや映画も、登場人物の区別がつかずにあまり楽しめません。
私は病気なのでしょうか？
'''

a = get_answer(q)
print(a)
```

```
 病気でしょう。そして、顔は病気ではないからです。どんな病気にも人根は信用できることです。
病気かどうかはわかりませんが、うれしいものです。顔は自然に消えていくことによって病気が徐々に消がされるようになります。そうすればどんな病気にも人根は信用できることが大切になります。
精神科を受診してください。
精神科で治療を受ければ今の悩みは解消に向かうのですが、それは顔の人根がない人根であることは、質問者心のことです。
この症状を本人も悩んでおられる方はすべています。
精神科を受診してください。
```

文全体が不自然です。病名の指摘がありません。もっとも、相貌失認自体症例数（学習データ数）が少ないため診断はもともと難しいのでしょう。

# 結論

このモデルを精神科の診断、治療に活かすのは難しいと思います。

1. 病気を正しく診断できていない。
2. （おそらく）プロンプト、コンプリーションのトークン数に対して学習レコード数が少ないため学習が進んでいない。データの複雑さに対して、それを包絡できるだけのデータ数がない。

数十万レコードあれば、もしかすると精密な診断の可能性はあるかも知れません。以上。
