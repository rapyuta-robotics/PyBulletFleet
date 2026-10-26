# Fleet Scaling — Implementation Plan

**Date:** 2026-10-26
**Status:** Draft (spec.md の Scope Approval 後に確定)
**Spec:** [spec.md](spec.md)

**Goal:**
- コア単体: 1000 台・headless・衝突判定ありで RTF 50x 以上 (dt=0.1 で 2.0 ms/step 以下)。
  10000 台では RTF 5x 以上 (20 ms/step 以下)。
- ROS ブリッジ込み: 1000 台、FleetState を sim time 10 Hz で publish し、RTF 10x 以上。
- 基準ホスト: WSL2 (16 cores、rendering なし)。判定は 3 回計測の中央値。

**Approach:** 状態の正本を SoA 配列に移し、ステップ単位の numpy カーネルで処理する。
PyBullet は遅延同期のミラーにする。コストの大きいフェーズから順に置き換え、各 PR の後に
定常状態ベンチマークで効果を確認する。

---

## 予算 (1000 台, 2.0 ms/step)

| フェーズ | 現状 | 目標 | 根拠 |
|---|---|---|---|
| collision (broad + narrow) | 18-19 ms | ≤ 0.8 ms | プロトタイプ 0.6-1.1 ms |
| AABB / grid 更新 | 5 ms | ≤ 0.1 ms | AABB を pose から配列計算すれば、grid は broad phase 内で毎回作り直せる |
| advance + Agent 更新 | 8 ms | ≤ 0.5 ms | カーネル 0.1-0.2 ms + イベント処理 |
| pose flush (PyBullet) | 2 ms | ≈ 0 ms (headless) | 遅延ミラー |
| step の固定オーバーヘッド | 約 0.1 ms | ≤ 0.2 ms | `getConnectionInfo`、イベント emit など |
| **合計** | **35-40 ms** | **≤ 2.0 ms** | |

予算に余裕がないのは advance + Agent 更新である。Python の `Agent.update()` は 1 回約 1.4 µs
なので、1000 台全員を毎ステップ呼ぶだけで 1.4 ms を使い切る。したがって P3 (イベント駆動化) は必須とする。

## 予算 (ROS ブリッジ込み, 1000 台, RTF 10x, FleetState 毎ステップ)

dt=0.1 で RTF 10x なので、wall time の予算は 10 ms/step である。

| 項目 | 現状 | 目標 |
|---|---|---|
| step (コア) | 35-40 ms | ≤ 2 ms (上の予算) |
| FleetState 生成 (`get_states`) | 約 6 ms | ≤ 1 ms (SoA から一括生成) |
| ROS メッセージ構築 + publish | 約 3 ms + DDS | ≤ 4 ms |
| 合計 | 約 45 ms (2x 相当) | ≤ 7 ms (余裕 3 ms) |

## 予算 (10000 台, RTF 5x, コア単体)

| 項目 | 現状 | 目標 | 根拠 |
|---|---|---|---|
| カーネル (advance + AABB + broad/narrow) | 約 395 ms | ≤ 5 ms | numpy プロトタイプ 4.6 ms |
| PyBullet ミラー | 31 ms | ≈ 0 ms | P2 |
| Python のエージェント処理 | 99 ms (phase1 の大半) | ≤ 5 ms | P3 でイベント数に比例する処理にする |
| 合計 | 426 ms | ≤ 20 ms | |

---

## PR 分割

各 PR は単独でマージでき、定常状態ベンチマークの数値を改善する。
分割の理由は、コア抽象の変更 (P1) と、挙動が変わりうる変更 (P4, P5) を
別々にレビューできるようにするためである。

### P0. 定常状態ベンチマークと計測基盤

- `prototype/steady_bench.py` を `benchmark/` の正式なシナリオとして追加する
  (`run_benchmark.py --type steady` など、既存 CLI に合わせる)。1000 台と 10000 台を計測対象にする。
- 計測規約: 基準ホストは WSL2、warmup 20 steps、計測 150 steps × 3 回、中央値を採用。
- `docs/benchmarking/results.md` に現状のベースライン (spec の計測表) を記録する。
- **効果:** なし (計測の土台)。
- **検証:** 3 回計測のばらつきを記録する。

### P1. SoA state store と配列ベースの衝突 broad phase

- core が所有する `StateStore` (pos, orn, aabb, footprint, flags, object_id ↔ row の対応) を新設する。
  行の追加・削除は、既存 batch controller と同じ「末尾行と入れ替えて詰める」方式にする。
- `SimObject` の pose キャッシュを store の行へのアクセサに置き換える。`get_pose()` / `set_pose()` の挙動は変えない。
- `set_poses()` は配列のまま store に書き込む (per-object ループをなくす)。
- 単一ボディのオブジェクトは、生成時に一度だけ `getAABB` でローカル AABB を取り、
  以降は pose から配列で AABB を計算する。多リンク URDF は従来どおり `getAABB` を使う。
- broad phase: セルキーのソートと `searchsorted` で候補ペアを作る
  (`prototype/soa_kernel.py` の方式)。moved-only の差分チェック、static の除外、
  `CollisionMode` (2D/3D、DISABLED) の扱いは、配列のマスクで再現する。
- narrow phase は、この段階では従来どおり PyBullet (`getClosestPoints` / `getContactPoints`)。
- `_cached_aabbs_dict`、`_cached_spatial_grid`、`_cached_object_to_cell` を削除する。
- **効果見込み:** collision と AABB/grid 更新で 23 ms → 2 ms 前後。
- **検証:** 衝突 parity テスト (同じシーンで、現行実装と同じ active pair 集合になること)。
  `tests/` の collision 系テストがすべて通ること。
- **リスク:** 最大の PR になる。`SimObject` の pose 経路は two-phase step の契約
  (`docs/architecture/two-phase-step.md`) に関わるので、契約は変えずに内部の保存先だけを変える。

### P2. PyBullet 遅延ミラー

- store に `mirror_dirty` フラグを持たせる。Phase 2 の flush では、ミラーが必要な行だけを書き込む。
  ミラーが必要なのは、GUI 有効時、recorder 有効時、narrow phase で PyBullet を使うペア、
  多リンク・アタッチ・物理オブジェクトである。
- `body_id`、`sim_core.client` など PyBullet ハンドルを公開するアクセサや、
  `getAABB` を使う ROS サービス (`sim_services.py:279`) の前に、全件 flush を挟む。
- **効果見込み:** pose flush 2 ms → headless ではほぼ 0。
- **検証:** GUI あり・なしで同じ衝突結果になること。recorder の出力が変わらないこと。

### P3. batch controller の store 直結と Agent 更新のイベント駆動化

- batch controller のバッファ (`_pos_buf`, `_orn_buf`) を store の行に置き換え、
  `set_poses` 経由の書き戻しをなくす。
- `_apply_phase1` の速度計算を配列化する (`p.getEulerFromQuaternion` の per-agent 呼び出しをやめる)。
- `step_once()` の per-object ループを、「Python の更新が必要なオブジェクト」の集合だけに回す。
  その集合に入るのは、非移動アクション実行中、関節補間中、link アタッチあり、
  per-step プラグインあり、PRE/POST_UPDATE ハンドラ登録ありのいずれかに当てはまるオブジェクトである。
  `batch_base.py:334` のコメントと実装の食い違い (`_needs_update` が False にならない) もここで解消する。
- batch カーネルは到着した行を返し、`MoveAction` の完了判定はそのイベントで駆動する。
- **効果見込み:** 8 ms → 0.5 ms 以下。
- **検証:** アクション、BT、プラグイン、ROS ブリッジのテスト。batch と per-agent で同じ軌跡になること。

### P4. プリミティブ narrow phase (箱と円を既定にする)

- 円・箱 (2D OBB) 同士の距離判定を numpy で実装する。`collision_margin` と signed distance を扱う。
- 形状の割り当て: cylinder / sphere は円、box・mesh・単一ボディ URDF はローカル AABB から求めた箱。
- オブジェクトごとに `collision_shape: primitive | pybullet` を選べるようにし、既定は `primitive` とする。
  物理オブジェクトと多リンク URDF は常に `pybullet`。
- 既定値の変更で衝突判定の結果が変わる既存テストは、意図した変更かどうかを確認したうえで更新し、
  PR に差分の一覧を載せる。
- **効果見込み:** 候補ペアが多い密集シーンで、`getClosestPoints` の per-pair 呼び出し (約 1-2 µs/ペア) をなくす。
- **検証:** 箱・円の形状で PyBullet の `getClosestPoints` と判定が一致すること。
  mesh を箱で近似した場合の差を、テストシーンで文書化する。

### P5. ROS ブリッジの状態配信を共有メモリ経由の別プロセスにする

設計は spec の「ROS ブリッジ: 状態配信の分離 (Q4)」を参照。2 つの PR に分ける。
分ける理由は、P5a が ROS に依存しない core 側の変更で、P5b が ROS 側の変更だからである。

**P5a. 共有メモリのスナップショット (core 側、ROS 非依存)**
- `multiprocessing.shared_memory` 上のリングバッファに、毎ステップ SoA のスナップショットを書く
  publisher を core に追加する。seqlock で整合を取る。レイアウト (列名、dtype、ヘッダ) をバージョン付きで定義する。
- 配信モード `every_step` (既定、満杯なら sim が待つ) と `latest` (待たない) を実装する。
- Python の reader API も用意する。ROS を使わない Python のツールやビューアも、同じ reader で状態を読めるようにする。
- **効果見込み:** sim 側の書き込みコストは 1000 台で数十 µs、1 万台で約 0.2 ms。
- **検証:** reader が破損したスナップショットを読まないこと (書き込み中の読み取りを含むテスト)。
  `every_step` で欠番がないこと。`latest` で sim が待たないこと。

**P5b. ブリッジプロセス (ros2_bridge 側)**
- 共有メモリを読み、FleetState、`/clock`、TF、odom を publish する別ノード (別プロセス) を追加する。
  メッセージは配列から一括構築し、per-robot の Python オブジェクト生成を減らす。
- 現行の BridgeNode からは状態 publish を外し、services / actions だけを残す。
- `publish_rate` (sim time 基準) は両モードに適用する。
- **検証:** `ros2_bridge/PERFORMANCE.md` の手順で、1000 台、`every_step`、`publish_rate: 10`、
  RTF 10x 以上を確認する。`latest` モードで、ブリッジを起動しても sim のコア性能が
  ROS なしと比べて 5% 以内しか落ちないことを確認する。`docker/test_bridge_api.sh` の bridge 統合スモークテスト。
- **ブリッジを C++ にする判断:** rclpy で `every_step` の 1 回あたり publish コストが目標に届かない場合。
  sim 側は共有メモリのレイアウトにしか依存しないので、P5a はそのまま使える。
- **範囲外 (follow-up):** 1 万台の全台 FleetState を毎ステップ配信すること。manager 単位のストリームに分け、
  複数のブリッジプロセスで publish する構成を検討する。

### P6 (条件付き). ネイティブカーネル

- P1-P3 の後、10000 台で 10x 以上が目標に加わった場合、または 10000 台 / 5x で予算が足りないと
  計測で分かった場合に着手する。
- まず Numba (pip 依存のみ) で置き換える。プロトタイプでは 10000 台のカーネルが
  4.6 ms (numpy) から 0.29 ms (1 スレッド) / 0.17 ms (16 スレッド) になった。
  配布や ROS (bloom) の要件を確認したうえで、必要なら C++ (nanobind, ament_cmake 互換) か Rust (PyO3) を選ぶ。
- numpy 実装は、フォールバック兼テスト用オラクルとして残す。
- カーネルをネイティブにしても、マルチプロセス化や空間分割は 10000 台では不要である
  (1 スレッドで 0.3 ms)。その規模の律速は Python のオーケストレーションと ROS I/O になる。

---

## 判断ポイント

- **P1 の後:** 定常状態ベンチマークで collision + AABB が 2 ms 前後に下がっているか確認する。
  下がっていなければ、Python 側の固定オーバーヘッドを再プロファイルし、P2 以降の順序を見直す。
- **P3 の後:** 1000 台で 2.0 ms/step、10000 台で 20 ms/step を満たしているか確認する。
  満たしていれば P4、P5 は機能要件 (ROS、密集シーン) のために進める。
  満たしていなければ、P6 を前倒しするか検討する。
- **案 D (フルスクラッチ) の再検討:** spec の「D を再検討する条件」に該当した時点。

## 検証方針

- 各 PR: 関連テスト + 定常状態ベンチマーク (1000 台、3 回中央値) の before/after を PR に記載する。
- P1 と P3: アーキテクチャ変更なので、マージ前に `make verify` を実行し、独立レビューを受ける。
- P5: ROS 2 ブリッジのチェック (`AGENTS.md` の verification matrix と bridge スモークテスト)。

## リスク

- **two-phase step の契約:** P1、P2 は pose の保存先と同期タイミングを変える。
  callback 中の `p.getBasePositionAndOrientation` は、もともと前ステップの値を返す契約なので
  互換は保てるが、テストで確認する。
- **ベンチマークのぶれ:** WSL2 上で ±10% 以上ぶれる。判定は中央値と相対比で行う。
- **プリミティブ近似による判定差:** 既定を箱・円にするため、mesh 形状のロボットでは判定が変わる。
  `collision_shape: pybullet` で従来の判定に戻せるようにする。
- **規模の上限:** numpy のままでは 5000 台で約 2.2 ms、1 万台で約 4.6 ms。
  1 万台で 10x を超える目標には P6、50x には案 D の検討が必要になる。

## Prototype

- `prototype/steady_bench.py`: 現行コアの定常状態ベンチマーク
  (`python3 docs/design/2026-10-26-fleet-scaling/prototype/steady_bench.py 1000 diff 60`)。
- `prototype/soa_kernel.py`: 同じ負荷の SoA + numpy カーネル
  (`python3 docs/design/2026-10-26-fleet-scaling/prototype/soa_kernel.py 1000`)。
  円近似の narrow phase を使うため、PyBullet のメッシュ判定とは厳密には一致しない。
- `prototype/numba_kernel.py`: 同じ負荷の Numba 版 (counting sort のグリッド + 並列の近傍探索)。
  Numba はリポジトリの依存に入っていないので、別の venv で実行する
  (`NUMBA_NUM_THREADS=1` で 1 スレッドの値を測れる)。候補ペア数が numpy 版と一致することを確認済み。
