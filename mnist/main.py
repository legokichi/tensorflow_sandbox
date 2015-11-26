# -*- coding : utf-8 -*-
# coding: utf-8
import tensorflow as tf
import input_data

# mnistデータを取得）
mnist = input_data.read_data_sets("/Users/yohsukeino/GitHub/Clone/mnist/", one_hot=True)

# 入力情報を持たせるためにplaceholderを定義する。
# shape(第2引数)の次元にNoneを指定すると、どんな長さの次元数であっても対応できる。
x = tf.placeholder("float", [None, 784])

# 重みとバイアスを保持するVariableを定義する。
# Variableは操作によって値を修正することのできる変数。
W = tf.Variable(tf.zeros([784, 10])) # 重み 784次元の入力を受けて10次元の出力を返す
b = tf.Variable(tf.zeros([10])) # バイアス 10次元の出力に加えられる

# ニューラルネットモデルを定義する。
# 入力xと重みWの行列積(tf.matmul)の出力にバイアスbを加え、ソフトマックスで最終的な出力を決定する。
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 以上でニューラルネットモデルの定義は終わり。
# 以下は、訓練・評価の実装
# (今回のサンプルでは、交差エントロピーで評価して学習しています。)

# 教師データ(正しい答え)を保持するplaceholderを定義する。
y_ = tf.placeholder("float", [None, 10])

# 交差エントロピーの計算式を定義する。
# 教師データy_とモデルからの出力yの対数をとったものとの積を取り、全体の合計を計算する
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 各ステップにおけるモデルの更新方法を定義する。
# 1ステップごとに0.01の更新率で交差エントロピーが最小になるようにする。
# (ニューラルネットの学習には誤差逆伝搬法が用いられるが、これはモデルが何かで判定しているらしい。)
# (ここでは、y = tf.nn.softmax(...)としたので誤差逆伝搬法がニューラルネットの更新に採用されるようだ。)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# すべての変数を初期化するための準備をする。
init = tf.initialize_all_variables()

# Sessionを定義し、すべての変数を初期化する。
sess = tf.Session()
sess.run(init)

# 訓練を行う
# mnistデータをSessionに渡して訓練を行い、train_stepの定義に合わせてモデルを更新する。
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 性能評価のための評価式を定義する。
# 入力データに対するモデルの出力yと教師データy_が一致しているか確認する。
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 最終的な性能評価は平均値で決定することを定義する。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# テストデータとそのラベルを使って性能を評価する。
print "score:" + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
