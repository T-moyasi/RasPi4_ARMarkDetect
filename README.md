Rasberry Pi4＋カメラモジュール環境でPython3+OpenCV3_Contlibを使って、
5つのARマーカを貼り付けた板型オブジェクトを検出します。
中心位置座標と角度、面積の計算をしています。
中心座標がx,y,基準面積からの差分がzと想定しています。
z-axis flipping問題の対策は多数決法を採用していますが、箱の表面にマーカを貼る想定ならばピッチとロールを0～180で正規化した方が正確です。

活用例としてドローン向け板状コントローラーや、無人販売所での商品の動き検出などが考えられます。
ARマーカ自体の情報は活用していないため、それらとの組み合わせで数多くのアプリケーション実装が考えられます。
