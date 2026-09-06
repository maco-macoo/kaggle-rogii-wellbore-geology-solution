# ROGII Wellbore Geology Prediction — 472nd-place solution

|  |  |
|---|---|
| **Competition** | [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) (Kaggle) |
| **Final rank** | 472 / 6,125 teams — bronze medal |
| **Final score** | private RMSE 8.939 (public 7.283) |
| **Submission** | Kaggle notebook (code competition: ≤ 9 h, no internet) |
| **Code** | [`solution.ipynb`](solution.ipynb) (EN) · [`solution.ja.ipynb`](solution.ja.ipynb) (JA) |

*日本語版はページ末尾。*

---

## Summary

Predict the geology along horizontal oil wells: for every 1-ft step of a wellbore, estimate
`tvt` — the well's vertical position within the stratigraphic column — from the trajectory,
the gamma-ray (GR) log, and a *typewell* reference that gives GR as a function of TVT.
Metric is plain RMSE in feet.

The solution treats this as classic geosteering inference. A **particle filter** matches the
horizontal GR log against the typewell curve; a **bidirectional GRU** trained on the
filter's posterior refines it; a **per-well robust polynomial projection** smooths the
result in stratigraphic coordinates; and a **spatial prior** interpolates the geology
directly from neighboring training wells where they are dense enough. Every hyperparameter
was chosen on leak-free by-well cross-validation — the public leaderboard, whose endgame was
dominated by probing-based forks that did not carry over to the private split, was never
used for tuning.

## Competition setup

- **Data**: 773 training wells. Each well is a trajectory (MD/X/Y/Z), a GR log, and a
  typewell (a vertical GR-vs-TVT reference for that location). The target `tvt` is scored on
  the evaluation zone — the part of the well beyond the last known TVT value.
- **Structure**: within a rock layer `dtvt ≈ −dz`; the hard part is the discrete global
  offset (which layer you are in) and the dip breaks where layers bend.
- **Code competition**: submissions are Kaggle notebooks re-run on a hidden test set of
  several hundred wells (9-hour limit, internet disabled). The public leaderboard scores 26%
  of the hidden test; the private leaderboard, and the final standings, use the other 74%.

## Approach

### 1. Particle filter (the backbone)

Per well, each particle advances a TVT hypothesis using the `dtvt ≈ −dz` structure with
dip-break dynamics, weighted by how well the observed GR matches the typewell's GR at the
hypothesized TVT (Cauchy likelihood). We run **128 seeds × 500 particles** and take the
likelihood-weighted posterior mean, ensembled across seeds. This alone reaches ~8.6 on the
public leaderboard — and the posterior is deliberately kept, not just its mean, because the
whole rest of the pipeline feeds on it.

### 2. BiGRU residual refiner

A bidirectional GRU consumes the well as a sequence on a 4-ft MD grid — ~36 channels of
particle-filter posterior statistics (seed paths, likelihoods), GR, trajectory, the typewell
grid, and a spatial-prior feature — and predicts the *residual* of the PF ensemble against
the truth. 5 folds × 3 seeds = **15 models, averaged**. At inference the kernel re-runs the
full 128-seed PF on the hidden test with a shared cache to fit the 9-hour budget.
PF + GRU: public **7.656**. (A GBDT stack over ~195 handcrafted features provides the
base prediction the GRU replaces — it survives only as a fallback on wells the leg
cannot cover.)

### 3. U-space projection

Well predictions should be smooth in stratigraphic coordinates. Per well, transform to
`U = pred + z` (anchored at the last known point), fit a **robust polynomial of degree 4**
over normalized measured depth (IRLS with Cauchy weights on a MAD scale), and pull the
prediction 75% of the way toward the fit. Public 7.656 → 7.641, and it stacks with
everything below; raising the degree from 3 to 4 was worth another −0.094 at the end.

### 4. Spatial prior for well-covered wells

Where a test well sits inside a dense cluster of training wells, the geology can be
interpolated directly, using the structural identity `TVT = surface(X, Y) − Z + b_well`:
the stratigraphic surface is reconstructed by inverse-distance weighting (k=20 neighbors)
over row-level point clouds of neighboring training wells, and the per-well offset comes
from the known prefix (leak-free). Applied only to wells passing three guards (≥ 30
known-prefix rows, well-conditioned neighbors, prefix residual ≤ 20 ft), as an outer blend
with weight 0.30. Public 7.641 → **7.377**, the single biggest lever after the backbone;
with degree-4 projection, **7.283**.

### 5. Test-time recoveries

Two conservative post-processes, each gated by self-validation on the visible prefix:

- **Duplicate-well recovery**: the hidden test contains byte-identical copies of training
  wells (including renamed ones — found by matching Z, GR, and TVT signatures). The
  organizers had re-issued labels, so a pure copy of the training truth hurts; the applied
  value is a 50/50 hedge between the transferred truth and the model prediction.
- **Contact override**: rebuild TVT directly from a formation-contact elevation
  (`tvt = ref_tvt − (Z − contact)`), replacing predictions only on wells where the
  reconstruction reproduces the known prefix within tolerance.

## Files

| File | Contents |
|---|---|
| [`solution.ipynb`](solution.ipynb) | The inference pipeline, annotated in English: a markdown cell explains each stage (core engine, spatial leg, GRU leg, recoveries, runner). |
| [`solution.ja.ipynb`](solution.ja.ipynb) | The same notebook with the annotations in Japanese. Code cells are identical. |
| [`train/`](train/) | Everything that produces the pretrained artifacts the notebook loads: `build_features.py` → `train_gbdt.py` (the GBDT stack) and `pf_dump.py` → `gru_features.py` → `gru_train.py` (the 15 GRU models), plus the shared `spatial_leg.py` / `pf_candidates.py` modules. |
| `LICENSE` | MIT. |

The code is the submitted pipeline cleaned up for publication — comments and internal
scaffolding were rewritten and dead fallback code removed; the algorithms and every frozen
parameter are unchanged.

## Reproduce

Training (competition data under `input/{train,test}`; GPU recommended):

```bash
python train/build_features.py                  # per-row feature tables
python train/train_gbdt.py                      # GBDT-PP artifacts -> train/artifacts/gbdt/
python train/pf_dump.py                         # 128-seed PF dumps for the 773 wells
python train/gru_features.py                    # 36-channel sequence tensors
for f in 0 1 2 3 4; do for s in 42 43 44; do    # 15 GRU models -> train/artifacts/gru/weights/
  python train/gru_train.py --fold $f --seed $s
done; done
```

Inference: run `solution.ipynb` top to bottom. On Kaggle, attach the competition dataset
plus the artifacts above as datasets; the notebook recomputes everything else — the
128-seed particle filter included — inside the 9-hour, no-internet rerun budget.

## License

MIT. See [`LICENSE`](LICENSE).

<br>

---

<br>

# 日本語

# ROGII Wellbore Geology Prediction · 472位の解法

|  |  |
|---|---|
| **コンペ** | [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)（Kaggle） |
| **最終順位** | 6,125チーム中 472位 = 銅メダル |
| **最終スコア** | private RMSE 8.939（public 7.283） |
| **提出形式** | Kaggle ノートブック（コードコンペ: 9時間以内・インターネット不可） |
| **コード** | [`solution.ipynb`](solution.ipynb)（英） · [`solution.ja.ipynb`](solution.ja.ipynb)（日） |

## 要約

水平坑井に沿った地質を予測するコンペである。坑井の 1ft ごとの各行について、軌道・ガンマ線（GR）
検層・typewell（その地点の GR を TVT の関数として持つ縦方向リファレンス）から、ターゲット `tvt`
＝地層に対する坑井の縦方向位置を推定する。評価指標は素の RMSE（ft）。

解法は古典的なジオステアリング推論である。水平坑井の GR ログを typewell 曲線と照合する
**粒子フィルタ**を土台に、その事後分布で学習した**双方向 GRU** が精錬し、**井ごとのロバスト
多項式射影**が地層座標系で滑らかにし、訓練井が密な場所では**空間事前分布**が地質を直接補間する。
ハイパーパラメータはすべてリークのない井別クロスバリデーションで決めた — 終盤がプロービング系
フォークに支配され private に持ち越されなかった public リーダーボードは、チューニングには一切
使っていない。

## コンペ概要

- **データ**: 訓練 773 井。各井は軌道（MD/X/Y/Z）+ GR ログ + typewell を持つ。ターゲット `tvt`
  は評価ゾーン（TVT の既知値が尽きた先の区間）で採点される。
- **構造**: 地層内では `dtvt ≈ −dz`。難所は「どの層にいるか」という離散的なグローバルオフセットと、
  層が折れ曲がる dip の変化点である。
- **コードコンペ**: 提出は Kaggle ノートブックで、数百井の隠れテストに対して再実行される
  （9時間制限・インターネット不可）。public リーダーボードは隠れテストの 26%、private
  （＝最終順位）は残りの 74% で採点される。

## 解法

### 1. 粒子フィルタ（土台）

井ごとに、各粒子は `dtvt ≈ −dz` の構造 + dip 変化のダイナミクスで TVT 仮説を前進させ、
「仮説 TVT における typewell の GR」と観測 GR の一致度（Cauchy 尤度）で重み付けされる。
**128 シード × 500 粒子**を走らせ、尤度加重の事後平均をシード間でアンサンブルする。これだけで
public ~8.6。事後分布そのものを捨てずに保持するのがポイントで、以降のパイプライン全体が
この事後分布を食べて動く。

### 2. 双方向 GRU による残差精錬

坑井を 4ft 刻みの MD グリッド系列として双方向 GRU に入れる。チャネル（約36本）は粒子フィルタの
事後統計（シード経路・尤度）・GR・軌道・typewell グリッド・空間事前特徴で、予測するのは
PF アンサンブルの真値に対する**残差**。5 fold × 3 シード = **15 モデルの平均**。推論時は
カーネルが隠れテストに対して 128 シードの PF を共有キャッシュ付きで再実行し、9時間制限に収める。
PF + GRU で public **7.656**。（約195の手作り特徴量による GBDT スタックは GRU が置き換える
ベース予測を供給する — GRU が予測を出せない井のフォールバックとしてのみ残る。）

### 3. U空間射影

坑井の予測は地層座標系では滑らかであるはずだ。井ごとに `U = pred + z`（最後の既知点でアンカー）
へ変換し、正規化 MD の上で**4次のロバスト多項式**（Cauchy 重みの IRLS・MAD スケール）を
フィットして、予測をフィットへ 75% 引き寄せる。public 7.656 → 7.641。以降の全部品と加算的に
効き、最後に次数を 3→4 に上げるだけでさらに −0.094 稼いだ。

### 4. 訓練井が密な井への空間事前分布

テスト井が訓練井の密な群れの中にあるなら、構造恒等式 `TVT = 地層面(X, Y) − Z + b_well` で
地質を直接補間できる。地層面は近傍訓練井の行レベル点群から逆距離加重（近傍 k=20）で復元し、
井ごとのオフセットは既知区間から決める（リークなし）。3つのガード（既知区間 ≥30行・近傍の
条件数・既知区間残差 ≤20ft）を通る井だけに、重み 0.30 の外側ブレンドとして適用する。
public 7.641 → **7.377** と、土台以降では最大のレバーになった。4次射影と合わせて **7.283**。

### 5. テスト時のリカバリー

可視区間での自己検証をゲートにした、保守的な後処理を2つ。

- **重複井リカバリー**: 隠れテストには訓練井のバイト同一コピー（改名されたものを含む —
  Z・GR・TVT の署名照合で発見）が存在する。ただし運営がラベルを再発行しているため真値の
  純コピーは悪化する。適用値は「転写した真値」と「モデル予測」の 50/50 ヘッジ。
- **コンタクトオーバーライド**: 地層コンタクトの高度から TVT を直接再構築し
  （`tvt = ref_tvt − (Z − contact)`）、その再構築が既知区間を許容誤差内で再現する井に限って
  予測を置き換える。

## ファイル

| ファイル | 内容 |
|---|---|
| [`solution.ipynb`](solution.ipynb) | 推論パイプライン（英語解説付き）。各ステージ（コアエンジン・空間レッグ・GRU 脚・リカバリー・ランナー）の前に markdown セルで説明を入れてある。 |
| [`solution.ja.ipynb`](solution.ja.ipynb) | 同じノートブックの解説を日本語にしたもの。コードセルは同一。 |
| [`train/`](train/) | ノートブックが読み込む学習済みアーティファクトを作るすべて: `build_features.py` → `train_gbdt.py`（GBDT スタック）と `pf_dump.py` → `gru_features.py` → `gru_train.py`（GRU 15モデル）、および共有モジュール `spatial_leg.py` / `pf_candidates.py`。 |
| `LICENSE` | MIT。 |

コードは提出パイプラインを公開用に整理したもの — コメントと内部の足場を書き直し、使われて
いないフォールバックコードを除去してある。アルゴリズムと凍結パラメータはすべて提出時のまま。

## 再現

学習（コンペデータを `input/{train,test}` に配置。GPU 推奨）:

```bash
python train/build_features.py                  # 行単位の特徴量テーブル
python train/train_gbdt.py                      # GBDT-PP 成果物 -> train/artifacts/gbdt/
python train/pf_dump.py                         # 773 井の 128シード PF ダンプ
python train/gru_features.py                    # 36チャネル系列テンソル
for f in 0 1 2 3 4; do for s in 42 43 44; do    # GRU 15モデル -> train/artifacts/gru/weights/
  python train/gru_train.py --fold $f --seed $s
done; done
```

推論は `solution.ipynb` を上から実行する。Kaggle 上ではコンペデータと上記アーティファクトを
データセットとしてアタッチすれば、128シードの粒子フィルタを含む残りすべてを 9時間・
インターネット不可の再実行予算内で再計算する。

## ライセンス

MIT。[`LICENSE`](LICENSE) を参照。
