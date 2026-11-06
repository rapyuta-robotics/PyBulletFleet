# Fleet Scaling — Implementation Plan

**Date:** 2026-10-26 (2026-10-27 改訂: レビュー反映。2026-11-05 改訂: #94 後の基準値、P0 実装)
**Status:** Draft (spec.md の Scope Approval 後に確定)
**Spec:** [spec.md](spec.md)
**基準コミット:** upstream `6ee451e` (#94、#98 入り)。行番号付きリンクは `6551125` の permalink

**Goal:**
- コア単体: 1000 台・headless・衝突判定ありで RTF 50x 以上 (dt=0.1 で 2.0 ms/step 以下)。
  10000 台では RTF 5x 以上 (20 ms/step 以下)。
- ROS ブリッジ込み: 1000 台、FleetState を sim time 10 Hz で publish し、RTF 10x 以上。
- 基準ホスト: WSL2 (16 cores、rendering なし)。判定は 3 回計測の中央値。

**Approach:** 状態の正本を SoA 配列に移し、ステップ単位の numpy カーネルで処理する。
PyBullet は遅延同期のミラーにする。コストの大きいフェーズから順に置き換え、各 PR の後に
定常状態ベンチマークで効果を確認する。

**前提となる upstream PR:** #94 (broad phase) と #98 (`query_aabb`) は 2026-11-04 に、
#111 (NORMAL_2D の非推奨化) と #114 (軸ごとの collision clearance) は 2026-11-05 に、
#115 (多リンクボディの AABB を全リンクの和にする) は 2026-11-06 にマージ済み (main `2e2a86a`)。
#87 (PathRequest) を P3 の前にマージする。
P1 は #94 + #114 の意味論を、P3 は #87 の request 状態を、それぞれ前提にする
(spec の「関連する upstream PR」)。

---

## 予算 (1000 台, 2.0 ms/step, 疎シナリオ)

現状は `2e2a86a` (#115 入り) の 2026-11-06 の計測 (P0 の sparse シーン、3 回の中央値、他の負荷なし)。
P1 後 (`486cef8`) の値は P1 の節の表にある (sparse 3.3 ms: phase1 1.3、pose flush 1.1、AABB flush 0.3、
collision 0.5)。
2026-11-05 に `6ee451e` で測った 14.2 ms (phase 別 6.1 / 2.0 / 3.7 / 1.8) は、同じ日に他の作業が
走っていた状態の値で、同一セッションで `6ee451e` と `2e2a86a` を交互に測ると差はノイズ以下
(sparse 10.2-11.5 ms、carrying 30.7-31.7 ms がどちらにも出る)。つまり #106-#115 は step 時間を変えず、
日をまたいだ絶対値の比較は負荷の差を見ている。**リビジョン間の比較は同一セッションで交互に測る。**
#94 前 (`6551125`) の値を括弧に残す。

| フェーズ | 現状 | 目標 | 根拠 |
|---|---|---|---|
| collision (broad + narrow) | 1.4 ms (#94 前 18.7-19.6) | ≤ 0.8 ms | プロトタイプ 0.6-1.1 ms |
| AABB / grid 更新 | 2.9 ms (4.8-5.4) | ≤ 0.1 ms | AABB を pose から配列計算すれば、grid は broad phase 内で毎回作り直せる |
| advance + Agent 更新 | 4.8 ms (5.4-6.0) | ≤ 0.5 ms | カーネル 0.1-0.2 ms + イベント処理 |
| pose flush (PyBullet) | 1.5 ms (1.6-1.8) | ≈ 0 ms (headless) | 遅延ミラー |
| step の固定オーバーヘッド | 約 0.1 ms | ≤ 0.2 ms | `getConnectionInfo`、イベント emit など |
| **合計** | **11.1 ms (9x)** (30.5-32.8) | **≤ 2.0 ms** | 余裕 0.4 ms。プロトタイプにない static / mode 混在 / margin の処理をここで吸収する |

#94 の後は collision が最大のフェーズではない。残る 9.7 ms のうち、per-object の agent update が 4.8 ms、
AABB / grid と pose の flush が 4.4 ms で、どちらも「オブジェクト単位の Python 処理」である。
P1 (store と配列の flush) と P3 (per-object ループの除去) がこの 9.7 ms の対象になる。
予算に余裕がないのは advance + Agent 更新である。Python の `Agent.update()` は 1 回約 1.4 µs
なので、1000 台全員を毎ステップ呼ぶだけで 1.4 ms を使い切る。したがって P3 (イベント駆動化) は必須とする。

## 予算 (1000 台, 搬送シナリオ, 3.0 ms/step)

現状 (`2e2a86a`、2026-11-06、他の負荷なし): 27.0 ms (3.7x)。sparse の 2.4 倍で、子オブジェクト 1000 個が
per-object に処理されている (2026-11-05 の 50.1 ms は負荷のある状態の値)。

| フェーズ | 現状 | 目標 | 根拠 |
|---|---|---|---|
| advance + Agent 更新 + 子の pose 伝播 | 10.3 ms | ≤ 0.7 ms | 子の pose は `parent_row` / `attach_offset` 列からカーネルが配列で計算する (P1) |
| AABB / grid 更新 (親子 2000 体) | 6.8 ms | ≤ 0.2 ms | 配列計算 (P1) |
| collision | 5.7 ms | ≤ 1.0 ms | 親子ペアの無視は候補ペアのマスク (P1)。候補は sparse の約 2 倍 |
| pose flush (親子 2000 体) | 3.5 ms | ≈ 0 ms | 遅延ミラー (P2) |
| **合計** | **27.0 ms** | **≤ 3.0 ms** | 余裕 1.1 ms |

## 予算 (1000 台, 密集・交差シナリオ, 4.0 ms/step)

| フェーズ | 目標 | 根拠 |
|---|---|---|
| broad phase | ≤ 1.0 ms | 候補ペアは疎シナリオの数倍 (P0 で計測し、ここに記入する) |
| narrow phase (2D OBB / 円) | ≤ 1.5 ms | 候補ペア数に比例。OBB の SAT は円の数倍のコストなので、P0 で OBB 版のプロトタイプを測って確定する |
| 衝突イベント (開始 / 解消の差分、Python 配信) | ≤ 0.5 ms | ペア集合の差分は配列で取り、Python のハンドラ呼び出しは変化したペアだけ |
| その他 (疎シナリオと同じ) | ≤ 1.0 ms | |

## 予算 (ROS ブリッジ込み, 1000 台, RTF 10x, FleetState 毎ステップ)

dt=0.1 で RTF 10x なので、wall time の予算は 10 ms/step である。Q4 の構成では sim 側と
ブリッジ側が別プロセスで並列に走るので、`every_step` の定常状態の step 時間は
max(sim 側, ブリッジ側) になる。予算も 2 本に分ける。

**sim 側 (sim プロセスのスレッド):**

| 項目 | 現状 | 目標 |
|---|---|---|
| step (コア) | 14.2 ms (`6ee451e`) | ≤ 2 ms (上の予算) |
| スナップショット書き込み (SoA → 共有メモリ) | なし (現状は in-process で `get_states` 約 6 ms + 構築約 3 ms) | ≤ 0.2 ms |
| 合計 | 約 23 ms (4.3x 相当) | ≤ 2.5 ms |

**ブリッジ側 (ブリッジプロセス、1 publish あたり):**

| 項目 | 現状 (in-process) | 目標 |
|---|---|---|
| 状態の読み出し (`get_states` 相当) | 約 6 ms (Agent ごとに `Pose` を生成) | ≤ 0.5 ms (共有メモリの配列を読むだけ) |
| ROS メッセージ構築 | 約 3 ms | ≤ 4 ms (配列から一括構築) |
| publish + DDS | 未計測 (PERFORMANCE.md の計測に含まれる) | ≤ 3 ms |
| 合計 | 約 9 ms + DDS | ≤ 8 ms (余裕 2 ms) |

P5 の後、`get_states` のコストはブリッジ側に計上する (sim 側には残らない)。ブリッジ側の
合計が 10 ms を超えると `every_step` で sim が待ち、RTF はブリッジ側で頭打ちになる。
その場合の選択肢は、P5b の C++ 化か `latest` モードである。

## 予算 (10000 台, RTF 5x, コア単体)

現状は `9780808` の計測。phase1 (99 ms) は advance の配列計算とその周辺の Python 処理を
分けて測っていないので、ここでは全額を「Python のエージェント処理」に計上する。

| 項目 | 現状 | 目標 | 根拠 |
|---|---|---|---|
| カーネル (AABB + broad/narrow) | 約 296 ms (collision 231 + AABB/grid 65) | ≤ 5 ms | numpy プロトタイプ 4.6 ms (advance を含む) |
| PyBullet ミラー | 31 ms | ≈ 0 ms | P2 |
| Python のエージェント処理 (phase1 全体) | 99 ms | ≤ 5 ms | P3 でイベント数に比例する処理にする |
| 合計 | 426 ms | ≤ 20 ms (余裕 10 ms) | |

---

## PR 分割

各 PR は単独でマージでき、定常状態ベンチマークの数値を改善する。
分割の理由は、コア抽象の変更 (P1) と、挙動が変わりうる変更 (P4, P5) を
別々にレビューできるようにするためである。

### P0. 定常状態ベンチマークと計測基盤

**実装済み (2026-11-05、ブランチ `perf/p0-steady-benchmark`、upstream `6ee451e` の上):**
`benchmark/steady_bench.py` (ワーカー)、`run_benchmark.py --type steady` (`--scenario` / `--compare` /
`--sweep` / `--warmup`、反復の中央値で集計)、`make bench-steady`、`batch_perf.py` のヘルパー公開
(`make_sim` / `make_fleet` / `make_spawn_params` / `make_grid_params` / `make_manager`)、テスト 22 件、
`docs/benchmarking/results.md` の "Steady-State Scenarios"。

- シナリオ (全台を 200 m 先に送り、warmup 20 step の後の 150 step を測る。値は `6ee451e`、3 回の中央値):
  - **疎 (sparse):** 2 m 間隔のグリッドを全台が +x に平行移動する。AABB の重なりがほぼないので、
    broad phase とカーネルの固定費を測る。1000 台 14.2 ms (7.0x)。
  - **密集・交差 (dense):** 2 群が 1 m 間隔の行で対向してすれ違う (margin 0.05 m)。1 割のロボットは
    停止したまま障害物になる。`COLLISION_STARTED` / `COLLISION_ENDED` の数を計測窓で記録する。
    P4 の効果とイベント処理のコストはこのシナリオで測る。1000 台 13.0 ms (7.7x)、接触 610 件。
  - **搬送 (carrying):** 疎シナリオの全台がベースに 1 個ずつ箱をアタッチして移動する。倉庫での通常形
    (ロボットがビンを運ぶ) で、子オブジェクトの pose 伝播 (`_propagate_to_attached`、
    [sim_object.py#L1412](https://github.com/yuokamoto/PyBulletFleet/blob/6551125ec55a8a2861f0f18a0ad810f20e4f1b81/pybullet_fleet/sim_object.py#L1412))
    と子の flush・衝突判定で step が 3.5 倍になる。1000 台 50.1 ms (2.0x)。
  - **倉庫 (warehouse):** 200 台が、1 m グリッドの床タイル 23.5k とビン 8.7k (計 32,924 の静的な箱) の間の
    通路を走る。静的ボディとは重ならないので broad phase の静的除外と、per-object ループが静的ボディを
    舐めないことを測る。7.4 ms (13.6x)、#94 前は 16-17 ms、numpy プロトタイプ 0.8 ms。spawn 16.8 s。
  - **待機 (idle):** 1000 台が全台停止。0.45 ms。P1 の配列 broad phase がこれより遅くならないことを見る。
- 計測規約: 基準ホストは WSL2、warmup 20 steps、計測 150 steps × 3 回、中央値を採用。
  出力は per-phase の mean / median / p95 を出し、判定には 3 回の median を使う。
- **残り:** 10000 台の sparse を同じ基盤で測る (現状 426 ms は `9780808` の値)。2D OBB (SAT) の
  narrow phase を `prototype/soa_kernel.py` に追加し、密集シナリオでの 1 ペアあたりのコストを測って
  密集シナリオの予算表を確定する。
- **効果:** なし (計測の土台)。
- **検証:** 3 回計測のばらつきを記録する (sparse の p50 は ±0.3 ms、倉庫は ±1.9 ms)。

### P1. SoA state store と配列ベースの衝突 broad phase

**着手 (2026-11-06、ブランチ `perf/p1-collision-parity`、upstream `275b9f0` の上): parity テストの土台
(`64ebf4f`)。** `tests/collision_parity/` に、スクリプト化した 10 シーン (margin 内ですれ違う 2 列、
軸ごとの clearance と setter / `params` 経由の margin 変更、`CollisionMode` の変更、static の無視、
アタッチ / デタッチ、ignore filter、2 ステップに 1 回の衝突判定、`CONTACT_POINTS` と `HYBRID` の物理ボディ、
経路を走る 4 台のロボット) と、ステップごとに `COLLISION_STARTED` / `COLLISION_ENDED` のイベント、
observation の active ペア、`signed_distance` をオブジェクト名で記録するレコーダ、`275b9f0` で記録した
golden (`golden/*.json`、合計 100 KB) を置いた。`tests/test_collision_parity.py` は再記録と golden を比べ、
`DOCUMENTED_DIFFERENCES` にない差と、書いてあるのに出なくなった差の両方で失敗する。
再記録は `python -m tests.collision_parity.record` (`--check` で比較のみ)。2 回記録して同一 (物理シーン含む)、
10 シーンの記録は合計 1 秒未満。フルスイート 2988 passed。
golden が固定した現行の性質 (P1 で保つか、差として文書化する): DISABLED にしたボディの active ペアは
`COLLISION_ENDED` なしに消え、戻すと `COLLISION_STARTED` が改めて出る。その場で `detach_object()` した
carrier と load は接触したままなので、次の判定からペアとして報告される。
#115 のマージ (2026-11-06、`2e2a86a`) 後に golden を再記録した (`061a674`): 10 シーンはすべて同じ
(`agents_crossing` のロボットはベースで当たるので変わらない)。#115 の対象を押さえるため、アームがリンクで
箱に触れるシーン `link_contact` (上の箱に手先が接触、肘を曲げて上腕が横の箱に接触、関節補間で離れる) を
足した。このシーンは `275b9f0` では 3 ペアとも見つからない (29 差分)。golden は 11 シーン、参照は `2e2a86a`。
P0 の定常状態ベンチマークは `2e2a86a` で測り直した (上の予算表)。

**実装済み (2026-11-06、ブランチ `perf/p1-collision-store`、P1 parity ブランチの上、`49ffec6`): 衝突の列ストアと
配列 broad phase (P1 の前半)。** `pybullet_fleet/collision_store.py` に、全オブジェクトの AABB、ローカル箱、
collision mode、physics フラグを列で持つ `CollisionStore` と、その上の空間ハッシュを置いた。ハッシュは
パックしたセルキーのソート済み配列 2 本 (STATIC の行は変更時だけ差し替え、それ以外の行は 1 つでも動けば
ソートし直す) で、動いた行のセルをまとめて `searchsorted` し、配列の AABB 判定 (#94 の登録規則、#114 の
軸ごとの clearance、NORMAL_2D の z=0、HYBRID の物理ペアは 0、`ignore_static`、DISABLED) を通して object_id の
ペアを返す。運動学の単一ボディの AABB は pose から計算する (ローカル箱を生成時に identity 姿勢で 1 回読み、
|R| で変換する。PyBullet の `btTransformAabb` と同じで、球だけ extents を回さない。Phase 3 で動いた分を 1 回の
配列演算で出す)。物理ボディと多リンクは従来どおり PyBullet から読む。`_cached_aabbs_dict` /
`_cached_spatial_grid` / `_cached_object_to_cell` と per-object のグリッド更新は削除し、`sim.cached_aabb()` /
`sim.grid_cells_of()` を代わりに置いた。parity 11 シーンは同一、store の単体テスト 54 件 (総当たり比較、
増分同期と全再構築の一致)、フルスイート 3091 passed。

同一セッションで before / after を 2 周交互に測った結果 (P0 の定常状態ベンチ、中央値):

| シーン | before (`2e2a86a`) | after (`49ffec6`) | AABB flush | collision |
|---|---|---|---|---|
| sparse 1000 | 9.6 ms (10.5x) | 6.5 ms (15.4x) | 2.6 → 0.7 ms | 1.25 → 0.55 ms |
| dense 1000 | 8.6 ms (11.6x) | 6.1 ms (16.4x) | 2.1 → 0.6 ms | 1.37 → 0.63 ms |
| carrying 1000 | 27.7 ms (3.6x) | 15.3 ms (6.5x) | 7.0 → 1.5 ms | 5.7 → 1.2 ms |
| warehouse 200 + 33k 静止体 | 3.5 ms (28x) | 3.3 ms (31x) | 0.5 → 0.2 ms | 0.34 → 0.37 ms |
| idle 1000 | 0.43 ms | 0.43 ms | 0 | 0.01 → 0.02 ms |

collision は目標 (≤ 0.8 ms) に入った。AABB flush の残り 0.7 ms は、動いた object ごとに `_cached_pose` を
Python で集めて配列にする分で、pose のキャッシュを store の列にする (下の残り) まで消えない。
生成は object あたり約 25 µs 増えた (ローカル箱とコライダーのリンクを add 時に読む。倉庫シーンの spawn
10.8 → 11.7 s)。生成の支配項は従来からの `deepcopy` (ShapeParams、約 100 µs) と `createMultiBody` (約 95 µs) で、
P1 の対象外だが後で下げる余地がある。
**実装済み (2026-11-07、同ブランチ、`88872f6` と `486cef8`): pose の列化とアタッチの配列伝播 (P1 の後半)。**
全オブジェクトの pose を store の `pos` / `orn` 列に持ち、`SimObject` の pose キャッシュ (`get_pose()` が返す
`Pose`) はその行へのビューにした (列の再確保と削除時の行の入れ替えではビューを張り直す)。step 中の
`set_poses()` は、動いた行を `set_pose` と同じ判定 (距離² > 1e-12、四元数の符号は無視) で配列のまま求めて
書き込み、ベースにアタッチされた子は carrier の pose と `attach_pos` / `attach_orn` 列から段ごとに配列で運ぶ
(四元数の積は倍精度、符号は PyBullet の `getRotation` と同じ規則。`multiplyTransforms` は単精度なので差は約 1e-7)。
物理ボディ、link アタッチ、運動学でない子を持つ carrier、STATIC は従来の per-object 経路に回す。Phase 3 は列から
直接 AABB を出し、Phase 2 は pending 行の pose を一度にリストへ変換して PyBullet に渡す (配列の行を 1 つずつ
渡すと PyBullet 側の読み取りが遅い)。物理ボディは `stepSimulation` の後に行へ読み戻す。`Pose` は要素の値で比較
する (`pose.position == [..]` はビューでは例外になるので `list()` を挟む。CHANGELOG に記載)。
テスト `tests/test_pose_columns.py` 20 件、parity 11 シーン同一、フルスイート 3104 passed。

同一セッションで 2 周交互に測った結果 (P0 の定常状態ベンチ、中央値。before は前半 `49ffec6`):

| シーン | 前半後 (`49ffec6`) | 後半後 (`486cef8`) | phase1 | pose flush | AABB flush | collision |
|---|---|---|---|---|---|---|
| sparse 1000 | 7.2 ms (14.0x) | 3.3 ms (30.0x) | 4.4 → 1.3 ms | 1.4 → 1.1 ms | 0.7 → 0.3 ms | 0.6 → 0.5 ms |
| dense 1000 | 6.1 ms (16.4x) | 3.3 ms (30.2x) | 3.7 → 1.2 ms | 1.1 → 1.1 ms | 0.6 → 0.3 ms | 0.6 → 0.6 ms |
| carrying 1000 | 15.8 ms (6.3x) | 7.7 ms (13.1x) | 9.3 → 2.5 ms | 3.3 → 3.1 ms | 1.6 → 0.7 ms | 1.25 → 1.2 ms |
| warehouse 200 + 33k | 3.1 ms | 2.5 ms | 2.0 → 1.5 ms | 0.5 → 0.5 ms | 0.2 → 0.1 ms | 0.35 → 0.34 ms |
| idle 1000 | 0.45 ms | 0.43 ms | | | | |

ベースライン `2e2a86a` からは sparse 9.6 → 3.3 ms、carrying 27.7 → 7.7 ms。sparse の残り 3.3 ms の内訳は
phase1 1.3 (batch controller の advance 約 0.3 + per-object の `update()` ループ 0.4 + `set_poses` の
配列経路 0.3 + イベント)、pose flush 1.1 (1000 回の `resetBasePositionAndOrientation`、P2 の対象)、
AABB flush 0.3、collision 0.5 (narrow phase の Python ループ含む)。
**残り:** step 外の `set_poses()` は per-object のまま (PyBullet への書き込みが要る)。per-object の `set_pose`
経路はキャッシュが ndarray のビューになった分、1 回あたり約 1 µs 遅い (非 batch のコントローラで 1000 台だと
約 1 ms。P3 で per-object 経路自体を減らす)。衝突結果の object_id 配列化と `stop_on_collision` の配列消費は
未着手 (narrow phase が PyBullet のままなので P4 と一緒に)。多リンクと物理ボディの AABB は PyBullet から読む
まま。

- core が所有する `StateStore` (pos, orn, vel, aabb, local_aabb, footprint, mode, owner, parent_row,
  attach_offset, flags, object_id ↔ row の対応) を新設する。行の追加・削除は、既存 batch controller と同じ
  「末尾行と入れ替えて詰める」方式にする。**行番号は core の外に出さない。** イベント、
  `query_aabb()`、衝突結果、スナップショットはすべて `object_id` で返す。
- **所有者 (owner) 列:** `KINEMATIC` (store が正本、ミラーは遅延) と `PHYSICS` (PyBullet が正本。
  `stepSimulation` の後に `getBasePositionAndOrientation` / `getAABB` で store を更新し、ミラーは常に即時、
  flush 時の速度保持と HYBRID の margin 0 は現行どおり)。物理ボディの per-object ループは残し、予算の対象外とする。
- **ベースアタッチ:** 子の行は `parent_row` と `attach_offset` を持ち、カーネルが `advance()` の直後に
  親の pose から配列で計算する (親 → 子の順に並べる)。`ignore_attached_collision` は候補ペアのマスク
  (`parent[i] == j or parent[j] == i`) にする。link アタッチ (アームの手先など) は従来どおり `getLinkState` の
  FK で、親のミラーを即時同期する行として扱う。
- `SimObject` の pose キャッシュを store の行へのアクセサに置き換える。`get_pose()` / `set_pose()` の挙動は変えない。
- `set_poses()` は配列のまま store に書き込む (per-object ループをなくす)。
- 単一ボディのオブジェクトは、生成時に一度だけ `getAABB` でローカル AABB を取り、
  以降は pose から配列で AABB を計算する。多リンク URDF は従来どおり `getAABB` を使う
  (#115 が入れば全リンクの和。P1 は多リンクの AABB 取得を 1 か所に閉じ込め、#115 の
  `notify_geometry_changed()` を関節変更の入口として残す)。
- broad phase: セルキーのソートと `searchsorted` で候補ペアを作る
  (`prototype/soa_kernel.py` の方式)。登録範囲と探索は #94 + #114 と同じ
  (各軸をその軸の clearance の半分だけ広げた AABB が届く全セルに登録し、自セルだけ探索)。
  clearance は `collision_margin` のスカラーまたは (x, y, z) の三つ組 (#114) で、
  NORMAL_2D のボディを含むペアは z を 0 にし、HYBRID で物理ボディを含むペアは全軸 0 にする。
  NORMAL_2D は #111 で非推奨 (削除は #139) なので、P1 ではこの z=0 のマスク以外に
  NORMAL_2D 固有の経路を作らない。#114 で加わった「broad phase を経ずに再判定する active ペア
  (動いたペア、margin 変更後の全ペア) にも軸ごとの AABB 判定を先に掛ける」動作も再現する。
  セルより大きいボディ (コンベア、エレベータの床、壁) は届くセルごとにキーを繰り返して登録する
  (`soa_kernel.py` は中心セル 1 つ + 3×3 探索なので、このままでは大きなボディのペアを落とす)。
  static の除外、`CollisionMode` (3D、2D の z=0、DISABLED) の扱いは配列のマスクで再現する。
  **moved-only:** グリッドのキーは持続させ、動いた行だけキーを更新して近傍探索する
  (動いた割合が閾値を超えたら全件ソートし直す)。全台停止のシナリオで現行の約 0.37 ms を下回ること。
- `query_aabb()` (#98) は store の AABB 配列に対する重なり判定で提供する。
  「最後の step 以降に pose が書かれたオブジェクト」は dirty マスクで、DISABLED は mode マスクで扱う。
- narrow phase は、この段階では従来どおり PyBullet (`getClosestPoints` / `getContactPoints`)。
- 衝突結果 (active pair の集合と、その差分) を object_id の配列で持ち、
  `stop_on_collision` (#96) など controller 側の消費もその配列で行う。
- `_cached_aabbs_dict`、`_cached_spatial_grid`、`_cached_object_to_cell` を削除する。
- **効果見込み:** sparse で collision 1.8 + AABB/grid 3.7 → 1 ms 前後 (#94 後の値から)。
  搬送では子の伝播と AABB/grid の配列化で 50 → 20 ms 前後 (残りは P2、P3 で落とす)。
- **検証:** 衝突 parity テスト: スクリプト化したシーン (アタッチ/デタッチ、`CollisionMode` の変更、
  `set_collision_margin` のスカラーと (x, y, z) の三つ組、ignore filter、`physics: true` のボディを含む) で、
  現行 main (`275b9f0`、#94 + #114) と `COLLISION_STARTED` / `COLLISION_ENDED` のイベント列
  (ペアとステップ番号) と `signed_distance` が一致すること。比較対象の実装は P1 で消えるので、
  現行 main で記録したトレースを fixture (golden) として持ち、再記録のスクリプトを添える。
  「同じ結果」と「文書化した差」をテストで区別する (差はシーンごとの許容リストに書き、
  リストにある差が出なくなったらそれも失敗にする)。NORMAL_2D は #111 で非推奨
  (pytest は警告をエラーにする) なので、シーンでは NORMAL_3D + `(m, m, 0)` を使う。
  `query_aabb` parity テスト (`tests/test_query_aabb.py` がそのまま通ること)。
  `tests/` の collision 系テストがすべて通ること。
- **リスク:** 最大の PR になる。`SimObject` の pose 経路は two-phase step の契約
  (`docs/architecture/two-phase-step.md`) に関わるので、契約は変えずに内部の保存先だけを変える。

### P2. PyBullet 遅延ミラー

**実装済み (2026-11-07、ブランチ `perf/p1-collision-store`、P1 の上、`c9b4488`)。** `SimulationParams.mirror`
(`lazy` 既定 / `always`) を追加した。`lazy` では Phase 2 の flush は PyBullet が毎ステップ持つ必要のある行
(物理ボディ、多リンク (AABB を PyBullet から読む)、link アタッチの親) だけを書き、他の行は store に
`mirror_dirty` を立てて保留する。シミュレーション自身が PyBullet を読む前には自分で同期する: narrow phase は
判定するペアのボディを、`query_aabb` の live 判定と recorder は読む前に、`SimObject.body_id` と `sim.client` の
アクセサは返す前に保留分を書く。`sim.sync_mirror(object_ids=None)` を公開した。`mirror: always`、GUI、
`needs_mirror = True` を持つプラグインとコールバックは従来どおり毎ステップ全件を書く。step 外の `set_pose`
(即時経路) は PyBullet に直接書いて保留を消す。切断後の同期は no-op (切断後に `sim.client` を読む例が
あった)。`docs/architecture/two-phase-step.md` に「保持した body_id での PyBullet 直接呼び出しは
`mirror: always` でだけ対応」と書いた。ROS サービスの `getAABB` は `obj.body_id` 経由なので同期される。
テスト `tests/test_lazy_mirror.py` 13 件、parity 11 シーン同一、フルスイート 3117 passed。

同一セッションで 2 周交互に測った結果 (before は P1 後 `486cef8`):

| シーン | P1 後 | P2 後 (`c9b4488`) | pose flush |
|---|---|---|---|
| sparse 1000 | 3.4 ms (29.7x) | 2.2 ms (46x) | 1.2 → 0.06 ms |
| dense 1000 | 3.3 ms (30.6x) | 2.2 ms (46x) | 1.1 → 0.06 ms |
| carrying 1000 | 7.8 ms (12.8x) | 4.0 ms (25x) | 3.1 → 0.12 ms |
| warehouse 200 + 33k | 2.5 ms | 1.7 ms | 0.5 → 0.03 ms |
| idle 1000 | 0.44 ms | 0.44 ms | |

sparse の目標 2.0 ms (50x) まで残り 0.2 ms。残る内訳は phase1 1.3 (batch advance 約 0.3、per-object の
`update()` ループ 0.4、`set_poses` 0.3、イベント)、collision 0.5、AABB flush 0.3。P3 の対象。
**残り:** headless でプリミティブ形状の静的ボディを Bullet に作らない (起動時間、follow-up)。

- store に `mirror_dirty` フラグを持たせる。Phase 2 の flush では、ミラーが必要な行だけを書き込む。
  ミラーが必要なのは、GUI 有効時 (GUI のマウス選択は `rayTest` を使う:
  [core_simulation.py#L907](https://github.com/yuokamoto/PyBulletFleet/blob/6551125ec55a8a2861f0f18a0ad810f20e4f1b81/pybullet_fleet/core_simulation.py#L907))、
  recorder (`getCameraImage`) 有効時、narrow phase で PyBullet を使うペア、
  多リンク・アタッチ・物理オブジェクトである。
- **直接呼び出しの方針 (spec 要点 5):**
  - `SimObject.body_id` と `sim_core.client` のアクセサは、返す前に同期する
    (`body_id` はそのオブジェクト、`client` は全件)。
  - `sim_core.sync_mirror(object_ids=None)` を公開する。
  - `mirror: always` (設定) で現行どおり毎ステップ全件同期にできる。
  - プラグイン・コールバックは `needs_mirror = True` を宣言すれば、毎ステップ Phase 2 で同期される。
  - 保持した `body_id` で PyBullet を直接呼ぶ経路は、`mirror: always` 以外ではサポート外。
    `docs/architecture/two-phase-step.md` に書く。
- `getAABB` を使う ROS サービス
  ([sim_services.py#L279](https://github.com/yuokamoto/PyBulletFleet/blob/6551125ec55a8a2861f0f18a0ad810f20e4f1b81/ros2_bridge/pybullet_fleet_ros/pybullet_fleet_ros/sim_services.py#L279))
  は、`body_id` アクセサ経由なので同期される。store の AABB を返す API に置き換えてもよい。
- **効果見込み:** pose flush 2.0 ms (搬送 6.9 ms) → headless ではほぼ 0。
- **follow-up (この PR には含めない):** headless でプリミティブ形状の静的ボディは Bullet にボディを作らない
  (`mirror: never` の行)。倉庫型シーンの spawn 16-19 s (ほぼ `createMultiBody`) と RSS を下げる。
  GUI や `collision_shape: pybullet` が後から必要になった行は、そのときにボディを作る。
- **検証:** GUI あり・なし、`mirror: always` あり・なしで同じ衝突結果になること。
  recorder の出力が変わらないこと。callback 中の `p.getBasePositionAndOrientation` が
  `mirror: always` で前ステップの値を返すこと (現行契約)。

### P3. batch controller の store 直結と Agent 更新のイベント駆動化

**実装済み (2026-11-07、ブランチ `perf/p1-collision-store`、P2 の上): #87 に依らない前半。** step の per-object
ループは、`update()` を持つオブジェクト (Agent、Device) のリストを core が add / remove で保守し、それだけを
回す (倉庫型シーンの 33k 静的ボディを毎ステップ見なくなった)。batch 管理下で自分の仕事がない Agent (アクション
なし、動いている関節なし、アタッチなし、プラグインなし、エンティティイベントなし) は `update()` の先頭で
返る。Agent の速度と yaw rate は store の列 (`lin_vel` / `ang_vel`) で、`_current_velocity` は行のビュー、
`_current_angular_velocity` は 1 要素ビューのプロパティにし、batch controller は動いた行の速度を配列で書く
(per-agent のコピーループを削除)。`set_poses()` は行を配列で解決し、per-object 経路が要る行は store の列
(`kinematic`、`per_object_pose`) のマスクで決める。Phase 3 の AABB flush も id のループをやめた。
テスト `tests/test_update_loop.py` 6 件、parity 11 シーン同一、フルスイート 3123 passed、ブリッジ統合テスト PASSED。

同一セッションで 2 周交互に測った結果 (before は P2 後 `c9b4488`):

| シーン | P2 後 | P3 前半後 | phase1 | AABB flush |
|---|---|---|---|---|
| sparse 1000 | 2.2 ms (45x) | 1.19 ms (84x) | 1.24 → 0.50 ms | 0.31 → 0.14 ms |
| dense 1000 | 2.2 ms (46x) | 1.25 ms (80x) | 1.23 → 0.49 ms | 0.28 → 0.13 ms |
| carrying 1000 | 3.9 ms (25x) | 2.8 ms (36x) | 2.0 → 1.3 ms | 0.65 → 0.32 ms |
| warehouse 200 + 33k | 1.6 ms | 0.52 ms | 1.12 → 0.17 ms | 0.11 → 0.05 ms |
| idle 1000 | 0.45 ms (231x) | 0.17 ms (585x) | 0.41 → 0.14 ms | |

**1000 台・疎シナリオの目標 (2.0 ms、50x) に到達した** (ベースライン `2e2a86a` の 9.6 ms から 8 分の 1)。
sparse の残り 1.19 ms は phase1 0.50 (batch advance 約 0.3、`set_poses` 0.15、イベント)、collision 0.47
(narrow phase の Python ループ、active ペアの再判定、`_moved_this_step` の set 操作)、AABB flush 0.14。
carrying の残り 2.8 ms は phase1 1.3 (子 1000 体ぶんの行の計算と pending / moved の set 更新) と
collision 1.07 (親子ペアの除外は `_is_ignored_collision` の per-pair Python)。
**残り (#87 待ち):** 到着イベント (`MoveAction` の毎ステップ `execute()` ポーリングをやめる)、
`MoveAction` 経由で全台を動かす構成でも同じ値になること、関節補間の `joint_target_pending`、
`AgentPlugin.on_batch_update` と `BatteryPlugin` の配列化、`FleetStateProvider` の一括読み出し。

現状の phase1 (5.4-6.0 ms) の内訳 (`6551125`、1000 台 diff、batch): batch カーネル本体 0.27 ms、
`_apply_phase1` 3.85 ms (うち `set_poses` 3.2 ms、per-agent の速度コピーループ
[batch_base.py#L474](https://github.com/yuokamoto/PyBulletFleet/blob/6551125ec55a8a2861f0f18a0ad810f20e4f1b81/pybullet_fleet/controllers/batch_base.py#L474)
約 0.6 ms)、アクションを持たない 1000 台の `Agent.update()` ループ約 0.55 ms。
カーネルは問題でなく、周りのループが問題である。倉庫型シーンでは、per-object ループが動かない
33k の静的ボディまで毎ステップ `_needs_update` を見るので、200 台しか動いていなくても phase1 が
3.8-4.0 ms になる (spec の「倉庫型シーン」)。

- batch controller のバッファ (`_pos_buf`, `_orn_buf`, `_lin_vel_buf`, `_ang_vel_buf`) を store の行に
  置き換え、`set_poses` 経由の書き戻しと速度のコピーループをなくす。`Agent.velocity` /
  `_current_velocity` は store の行のビューにする。
- `step_once()` の per-object ループを、「Python の更新が必要なオブジェクト」の集合だけに回す。
  集合は core が持ち、`_needs_update` の変化時に出し入れする (静的ボディと batch 管理下の
  アクションなしエージェントは入らない)。全オブジェクトを走査しない。
  現状はこの集合が小さくならない要因が 3 つある: `MoveAction.execute()` が毎ステップ `agent.is_moving` を
  ポーリングする ([action.py#L321](https://github.com/yuokamoto/PyBulletFleet/blob/6551125ec55a8a2861f0f18a0ad810f20e4f1b81/pybullet_fleet/action.py#L321))、
  拒否されたパスの再試行も停止までポーリングする、`_update_kinematic_joints` が URDF ロボットの
  全関節を毎ステップ見る。アクション (BT、ROS の navigate) で動く構成では全台がこの集合に入るので、
  以下で外す。
  その集合に入るのは、非移動アクション実行中、関節補間中、link アタッチあり、
  per-agent プラグインあり (配列対応でないもの)、PRE/POST_UPDATE ハンドラ登録ありのいずれかに
  当てはまるオブジェクトである。
  `batch_base.py` のコメントと実装の食い違い (`_needs_update` が False にならない:
  [batch_base.py#L502](https://github.com/yuokamoto/PyBulletFleet/blob/6551125ec55a8a2861f0f18a0ad810f20e4f1b81/pybullet_fleet/controllers/batch_base.py#L502))
  もここで解消する。
- **到着イベント:** batch カーネルは到着した行 (`_row_finished`) を返し、controller はその request を
  #87 の `PathRequestState.FINISHED` にする。`MoveAction` は request の結末 (FINISHED / GONE / 拒否) を
  イベントで受けて終わり、毎ステップの `execute()` ポーリングをやめる。拒否されたパスの再試行も
  「停止」イベントで駆動する。
- **関節補間:** 関節に目標が残っている Agent だけを集合に入れる (`joint_target_pending` フラグ)。
  関節を毎ステップ走査しない。
- **プラグインの配列化:** `AgentPlugin` に任意の `on_batch_update(dt, rows)` を加える。
  実装していれば core が 1 回だけ呼び、実装していなければ従来どおり Agent ごとに `on_update()`。
  `BatteryPlugin` は `on_batch_update` を実装し、`soc` / `is_charging` を配列で持つ。
  `FleetStateProvider` が読む `is_moving`、速度、`battery_soc`、`is_charging` は、
  すべて store か配列プラグインの列から一括で読めるようにする (P5a のスナップショットもここから作る)。
- **効果見込み:** 6.1 ms (`6ee451e`、搬送では 18.1 ms) → 0.5 ms 以下 (カーネル 0.27 ms + イベント処理)。
  `MoveAction` 経由で全台を動かす構成でも同じ値になること。
- **検証:** アクション、BT、プラグイン、ROS ブリッジのテスト。batch と per-agent で同じ軌跡になること。
  疎シナリオを `MoveAction` 経由 (BT と同じ経路) で動かした場合も予算内であること。
  battery プラグインありの 1000 台で、`on_batch_update` の有無で `battery_soc` が一致すること。

### P4. プリミティブ narrow phase (箱と円を既定にする)

- 円・箱 (2D OBB) 同士の距離判定を numpy で実装する。`collision_margin` と signed distance を扱う。
  判定は「z 区間の重なり (z の clearance は `collision_margin` の z 成分。#114 以降の
  broad phase と同じ) AND xy 平面のプリミティブ距離 ≤ margin」とする。
- 形状の割り当て: cylinder / sphere は円、box・mesh・単一ボディ URDF はローカル AABB から求めた箱。
- オブジェクトごとに `collision_shape: primitive | pybullet` を選べるようにし、既定は `primitive` とする。
  物理オブジェクトと多リンク URDF は常に `pybullet`。
- 既定値の変更で衝突判定の結果が変わる既存テストは、意図した変更かどうかを確認したうえで更新し、
  PR に差分の一覧を載せる。
- **効果見込み:** 密集シナリオで、`getClosestPoints` の per-pair 呼び出し (約 1-2 µs/ペア) をなくす。
  数値目標は P0 で確定した密集シナリオの予算 (narrow phase ≤ 1.5 ms)。
- **検証:** 箱・円の形状で PyBullet の `getClosestPoints` と判定が一致すること (z 違いのケースを含む)。
  mesh を箱で近似した場合の差を、テストシーンで文書化する。

### PC. コマンドの受理と適用の分離 (ros2_bridge 側、in-process、P1-P5 と独立)

設計は spec の「コマンドの受理と適用の分離」を参照。小さな PR で、順序に関係なく最初に入れられる。

**実装済み (2026-11-05、ブランチ `perf/pc-accept-then-apply`、upstream `6ee451e` の上):**
`StepCommandQueue` (`ros2_bridge/.../command_queue.py`)、`FleetCommandDispatcher.accept()`、
`fleet_api.ack_mode` / `ack_timeout`、`<namespace>/command_acks`、テスト 20 件、ブリッジ README と PERFORMANCE.md。

- fleet services と topic (`navigate`、`stop`、`attach`、`execute_action`、`joint_command`) のコールバックは
  検証 (`accept()`: 名前解決とフレーム・配列長の検査) だけを行い、適用は sim スレッドに渡す。
  sim スレッドが step の合間 (pacing の sleep 中、一時停止中、ループ未起動) にいればコールバック内で即時適用し、
  step 中なら PRE_STEP の先頭でまとめて適用する (step ロックを PRE_STEP から POST_STEP まで保持する)。
- **既定は `ack_mode: applied`** (計画からの変更)。スモークテストで、対象物が無い `/fleet/attach` が
  `accepted` では「accepted」と返ることが分かった。適用時の拒否 (`set_goal_pose()` が `False`、attach 対象なし) を
  service の応答で受け取る既存クライアント (RMF アダプタも `rejected_names` を見る) の意味を変えないため、
  既定は適用結果を返す。待ちは step 中に届いた場合だけで、最大 1 step の wall time (`ack_timeout` 既定 1.0 s)。
  `accepted` は大量コマンド向けのオプションで、適用結果は両モードとも `command_acks` に publish する。
- `set_entity_state` も同じ経路。`reset_simulation`、`spawn_entity`、`delete_entity` はハンドラの登録・破棄を伴うので
  従来どおりコールバック内で実行する (follow-up)。per-robot のハンドラも対象外 (1000 台では使わない構成)。
- **計測 (`docker/test_fleet_scale.sh`、1000 台、5 リクエストの合計 ack):** target_rtf 1.0 で 0.79 → 0.62 s、
  最大速度で 2.23 → 0.67 s。最大速度での差 (3.3 倍) が GIL の競合の分。残る約 0.6 s は 1000 台ぶんの軌道生成
  (`set_goal_pose` 約 0.5 ms × 1000) そのもので、P1 / P3 で batch controller の配列化と一緒に下げる。
  hybrid モードの 8 s はエンドポイント数が原因なので、この PR では直らない (使わない構成として文書化済み)。
- **検証:** ブリッジ単体テスト 207 件 (Docker)、`docker/test_bridge_api.sh` PASSED。
  `test_bridge_launch` / `test_config_launch` の 2 件は main でも落ちる既存の問題 (rclpy の `Context.init()` 二重呼び出し)。
- **判断:** コマンドリング (下記 follow-up) の条件は「`accepted` モードで最大速度の ack が 10 ms を超える」に読み替える。
  `applied` の ack は軌道生成の時間を含むので、その短縮は P1 / P3 の対象。

### P5. ROS ブリッジの状態配信を共有メモリ経由の別プロセスにする

設計は spec の「ROS ブリッジ: 状態配信の分離 (Q4)」を参照。2 つの PR に分ける。
分ける理由は、P5a が ROS に依存しない core 側の変更で、P5b が ROS 側の変更だからである。

**P5a. 共有メモリのスナップショット (core 側、ROS 非依存)**
**実装済み (2026-11-05、ブランチ `perf/p5a-state-plane`、upstream `6ee451e` の上):**
`pybullet_fleet/state_plane.py` (`SharedStateWriter` / `SharedStateReader` / `StatePlanePublisher` /
`collect_fleet_frame`)、テスト 22 件 (別プロセスの reader を含む)、`benchmark/state_plane_bench.py`、
`docs/how-to/state-plane.md`。

- `multiprocessing.shared_memory` 上のリングバッファ (K スロット、既定 8) に、毎ステップ
  スナップショット (`seq`、`sim_time`、`step`、`generation`、`object_id`、pose、velocity、moving / charging /
  has_battery のフラグ、battery) を書く。レイアウトはヘッダの `version` 付き。行の集合か名前が変わるたびに
  `generation` を上げ、名前表 (64 バイト固定幅) を書き直す。reader は `generation` が変わったら名前表を読み直す。
- 同期: ヘッダの `head` / reader ごとの `tail` と heartbeat、スロットごとの seqlock (`seq_begin` / `seq_end`)。
  **スロットの引き渡しはセマフォではなく名前付きパイプ** (reader ごとに 1 本 + writer に 1 本) にした。
  `multiprocessing` のセマフォは無関係のプロセスから名前で attach できず、POSIX セマフォは標準ライブラリに
  ないため。パイプの `select` / `read` は OS の待ちで、syscall がメモリバリアを兼ねる点は同じ。
- 配信モード `every_step` (既定。満杯なら `wait_timeout` 既定 1.0 s まで待ち、超えたら WARNING 1 回で
  上書きを続け、追いつけば復帰) と `latest` (待たない)。reader 不在時は待たない。heartbeat が
  `2 × wait_timeout` 途絶えた reader は外し、その reader は次の read で再登録する。
  後から attach した reader は、その時点の最新スナップショットから読み始める。
- reader 側の `resource_tracker` 対策は `unregister()` ではなく「登録させずに attach」にした
  (3.10-3.12 は `resource_tracker.register` を一時的に無効化、3.13 以降は `track=False`)。
  `unregister()` は、writer と tracker を共有するプロセス (同一プロセス、spawn した子) で writer 側の
  unlink を壊すため。
- `collect_fleet_frame()` は現状 agent を 1 台ずつ走査する。行ごとに numpy へ代入していた形
  (1000 台で 3.3 ms) を、列ごとにリストで集めて 1 回で変換する形 (2 ms) にした (`d15b89f`)。残りは
  属性の読み出しそのもの (pose、velocity のコピー、battery plugin の検索) で、P3 で store の列から
  直接作る形に置き換えるまでこれ以上は下がらない。
- **計測 (`benchmark/state_plane_bench.py`):** 書き込み単体 1000 行 19 µs、10000 行 41 µs。別プロセスの
  reader 1 つ (every_step) 付きでは reader のコピー時間 (0.15 ms / 0.9 ms) に律速される。
- **検証 (済):** 書き込み中のスロットを返さないこと、コピー中に上書きされたら読み直すこと、`every_step` で欠番が
  ないこと (スレッドと別プロセスの両方)、`latest` で sim が待たないこと、遅い reader で `wait_timeout` 後に
  上書きに切り替わり追いつけば復帰すること、reader 不在と close 後に待たないこと、heartbeat 切れの除外と
  再登録、reader プロセスの終了後もセグメントが残ること、`generation` と名前表。
  **未:** macOS (Apple Silicon) での整合性テスト。
  macOS (Apple Silicon) で整合性テストを手動で 1 回通す。

**P5b. ブリッジプロセス (ros2_bridge 側)**

**実装済み (2026-11-05、ブランチ `perf/p5b-state-publisher`、P5a の上):** `state_plane` 設定節、
`state_publisher` ノード (`state_publisher_node.py`)、`FleetStateEncoder` (`fleet_state_cdr.py`)、
`snapshot_to_fleet_state_msg`、`bridge.launch.py state_publisher:=true`、
`docker/test_fleet_scale.sh --state-plane every_step|latest`、`docker/state_publish_profile.py`、テスト 11 件。

- **着手条件の計測:** rclpy で 1000 台の FleetState を構築 7.3 ms + シリアライズ 3.8 ms = 約 11 ms で、
  8 ms の閾値を超えた。ただし `Publisher.publish()` はシリアライズ済みの `bytes` も受け付け、その publish は
  0.01 ms だった。そこで **C++ ではなく、FleetState の CDR を配列から直接組み立てる** `FleetStateEncoder` にした。
  robot 名の長さだけで決まるバイト列のテンプレートを generation ごとに持ち、pose / velocity / フラグ / battery を
  numpy で所定オフセットに書く。rclpy の `serialize_message` とバイト単位で一致することをテストで確認
  (空、1 台、長さの違う名前、日本語名、1000 台)。1000 台で 0.05 ms。
- `state_plane.enabled` のとき、BridgeNode は `StatePlanePublisher` を install し、`/clock` と全体の `/fleet/states` を
  publish しない。`state_publisher` が共有メモリを読み、`/clock` (スナップショットの `sim_time`) と `/fleet/states` を
  publish する。manager 単位のストリーム、per-robot の topic / TF は従来どおり in-process (follow-up)。
  FleetState の型は変えていない。
- `publish_rate` (sim time 基準) は bridge 側が plane に書く頻度として適用する。`state_publisher` 側にも同名の
  パラメータがある。
- 計測で見つけて直したもの: compose が `ipc: host` なので、SIGKILL された bridge のセグメントがホストの
  `/dev/shm` に残り、次の `state_publisher` がそれに attach して止まる。writer は同名の古いセグメントを置き換え、
  reader は `is_stale()` で置き換えを検知して再 attach、スクリプトは起動前に削除する (P5a 側に cherry-pick 済み)。
- **計測 (`docker/test_fleet_scale.sh --robots 1000 --publish-rate 10 --target-rtf 0 --measure-rtf`、
  静止ワークロード、同一ホストの購読者 1 つ):**

  | 状態配信 | max RTF | 配信 |
  |---|---|---|
  | in-process (現行) | 4.3-4.9x | 全スナップショット |
  | `--state-plane every_step` | 16.7x → 29.0x (`collect_fleet_frame` 改善後) | 全スナップショット (1 publish 0.7 ms) |
  | `--state-plane latest` | 17.1x → 32.6x (同上) | 最新のみ (この速度では欠落なし) |

  この時点で律速は publish ではなく sim の step になった。
- **移動ワークロードの計測 (受け入れ条件 6 の条件、2026-11-05、`21c2419`):** scale checker の RTF 窓は
  コマンド検証の後に開くので、それまでの「max RTF」はすべて静止した fleet の値だった。
  `test_fleet_scale.sh --rtf-workload moving` は窓の直前に全台へ `--moving-distance` (既定 500 m、
  グリッドの行ごとに向きを交互) のゴールを `/fleet/navigate` で送り、全台が `is_moving` になってから
  窓を開き、窓の中の FleetState で移動中の割合が `--min-moving-fraction` (95%) を下回ったら FAIL にする。
  `docker/core_scale_rtf.py --moving-distance` が ROS なしで同じワークロードを走らせる。
  1000 台、timestep 0.1 s、publish_rate 10 (= 毎ステップ)、全台移動、衝突判定毎ステップ、窓 10 s wall:

  | 構成 | max RTF | step | コア単体との差 |
  |---|---|---|---|
  | コア単体 (ROS なし) | 5.66x | 17.7 ms | 基準 |
  | in-process (現行) | 2.71x | 36.9 ms | +19.2 ms (毎ステップの FleetState 構築 + シリアライズ) |
  | `--state-plane every_step` | 5.14x | 19.5 ms | +1.8 ms (1.10 倍) |
  | `--state-plane latest` | 5.03x | 19.9 ms | +2.2 ms (1.12 倍) |

  state plane の差分はほぼ `collect_fleet_frame` の 2 ms で、ROS ノードの常駐と共有メモリ書き込みは
  合わせて 0.1 ms 未満 (`enable_time_profiling` で `events_post_step` が step の 85%、それ以外は
  コア単体と同じ)。**受け入れ条件 6 の「1.05 倍以内」は未達 (1.10 倍)** で、残りの 2 ms は P3 で
  フレームを store の列のスライスにするまで消えない。「10x 以上」の方はコア単体が 5.7x なので、
  P1-P4 でコアの step が下がってから測り直す (ブリッジ側はもう律速ではない)。
- **検証 (済):** ブリッジ単体テスト 199 件 (Docker)、`docker/test_bridge_api.sh` PASSED (state plane なしの既定構成)、
  コア 2746 件、`make lint` / `make docs`。
  **未:** 受け入れ条件 6 の 3 回中央値・60 s 窓での正式計測 (コアの高速化後)。
- **ブリッジを C++ にする判断:** 不要になった。rclpy + `FleetStateEncoder` で 1 publish 0.7 ms なので、
  `every_step` のままでも 1000 台で 1500 publish/s 程度までは配信側が律速にならない。
  sim 側は共有メモリのレイアウトにしか依存しないので、必要になれば C++ の reader を後から足せる。
- **範囲外 (follow-up):** 1 万台の全台 FleetState を毎ステップ配信すること。manager 単位のストリームに分け、
  複数のブリッジプロセスで publish する構成を検討する。
- **条件付き follow-up: コマンドリング。** PC の後も `accepted` モードで最大速度の `/fleet/navigate` の ack が 10 ms を超える場合に、
  コマンドも共有メモリ (ブリッジ → sim の要求リングと、sim → ブリッジの応答リング) に移し、
  sim プロセスから ROS を外す。コストは spec の「コマンドの受理と適用の分離」に列挙したとおりで、
  in-process モードは残す。

### P6 (条件付き). ネイティブカーネル

- P1-P3 の後、10000 台で 10x 以上が目標に加わった場合、または 10000 台 / 5x で予算が足りないと
  計測で分かった場合に着手する。
- まず Numba (pip 依存のみ) で置き換える。プロトタイプでは 10000 台のカーネルが
  4.6 ms (numpy) から 0.29 ms (1 スレッド) / 0.17 ms (16 スレッド) になった。
  プロトタイプの密なグリッド (セル数 = nx × ny) は広いマップで膨らむので、採用時は
  `soa_kernel.py` と同じソートベースのハッシュにする。
  配布や ROS (bloom) の要件を確認したうえで、必要なら C++ (nanobind, ament_cmake 互換) か Rust (PyO3) を選ぶ。
- numpy 実装は、フォールバック兼テスト用オラクルとして残す。
- カーネルをネイティブにしても、マルチプロセス化や空間分割は 10000 台では不要である
  (1 スレッドで 0.3 ms)。その規模の律速は Python のオーケストレーションと ROS I/O になる。

---

## 判断ポイント

- **#94 マージ後 (P0 の途中):** ベースラインを取り直す。2026-11-05 に `6ee451e` で取り直し済み
  (sparse 14.2 ms)。P1 の目標値は変えない。
- **P0 の後:** 密集シナリオの OBB プロトタイプの数値で、narrow phase の予算 (≤ 1.5 ms) を確定する。
  届かなければ、円近似の適用範囲を広げる (箱の縦横比が 1.5 未満なら円) か、P6 を前倒しする。
- **P1 の後:** 疎シナリオで collision + AABB が 2 ms 前後に下がっているか確認する。
  下がっていなければ、Python 側の固定オーバーヘッドを再プロファイルし、P2 以降の順序を見直す。
- **P3 の後:** 1000 台で 2.0 ms/step、10000 台で 20 ms/step を満たしているか確認する。
  満たしていれば P4、P5 は機能要件 (ROS、密集シーン) のために進める。
  満たしていなければ、P6 を前倒しするか検討する。
- **PC の後:** `accepted` モードで最大速度の 1000 台の `/fleet/navigate` の ack が 10 ms を超えていれば、コマンドリングを
  follow-up から本計画に入れる (2026-11-05: `applied` 既定で計測済み。`accepted` の計測はスケールチェッカーに
  `ack_mode` を渡す口を足してから)。
- **P5b の後:** ブリッジ側の 1 publish が 8 ms を超えていれば C++ 化を判断する
  (2026-11-05: `FleetStateEncoder` で 0.6 ms。C++ 化は不要と判断)。
- **案 D (フルスクラッチ) の再検討:** spec の「D を再検討する条件」に該当した時点。

## 検証方針

- 各 PR: 関連テスト + 定常状態ベンチマーク (1000 台、疎と密集、3 回中央値) の before/after を PR に記載する。
- P1 と P3: アーキテクチャ変更なので、マージ前に `make verify` を実行し、独立レビューを受ける。
- P5: ROS 2 ブリッジのチェック (`AGENTS.md` の verification matrix と bridge スモークテスト)。
- `docs/` は pyright (`pyrightconfig.json` の include は `pybullet_fleet` と `tests`) と pytest
  (`testpaths = ["tests"]`) の対象外。black と flake8 (E402 は hook で無視) は対象なので、
  プロトタイプも pre-commit を通す。numba は依存にないので、`numba_kernel.py` は CI では実行しない。

## リスク

- **two-phase step の契約:** P1、P2 は pose の保存先と同期タイミングを変える。
  callback 中の `p.getBasePositionAndOrientation` は `mirror: always` では前ステップの値を返し
  (現行契約)、遅延ミラーでは最後に同期した値を返す。この違いは文書化し、直接呼び出しは
  アクセサ経由か `sync_mirror()` に誘導する。
- **upstream との競合:** P1 は #94 と同じ場所を変える。#94 を先に入れ、P1 はその上で書く。
  #98 の `query_aabb()` と #87 の `MoveAction` も同じ理由で先に入れる。#111 と #114 は
  2026-11-05 に入った。#115 (多リンクの AABB) も AABB のキャッシュを変えるので P1 の前に入れる。
- **ベンチマークのぶれ:** WSL2 上で ±10% 以上ぶれ、他の作業が走っていると同じコードで 1.5-1.9 倍遅く出る
  (2026-11-05 の 50.1 ms と 2026-11-06 の 27.0 ms は同じ carrying シーン)。判定は同一セッションで
  before / after を交互に測った中央値と相対比で行い、別の日の絶対値とは比べない。
- **プリミティブ近似による判定差:** 既定を箱・円にするため、mesh 形状のロボットでは判定が変わる。
  `collision_shape: pybullet` で従来の判定に戻せるようにする。
- **OBB narrow phase のコスト:** 円判定のプロトタイプしかない。P0 で OBB 版を測ってから予算を確定する。
- **プロトタイプと実装の差:** `soa_kernel.py` は毎ステップ全件を再ソートし (O(N))、ボディを中心セル 1 つに
  しか登録せず、アタッチも物理ボディもない。P1 の実装は moved-only、マルチセル登録、アタッチの伝播を
  持つので、プロトタイプの数値より重くなる。搬送・待機シナリオで差を測る。
- **rclpy の FleetState 構築:** per-robot のオブジェクト生成が下限 (約 2.8 ms) で、ブリッジ側の予算 8 ms の
  3 分の 1 を占める。P5b の着手条件で判定し、届かなければ C++。
- **共有メモリの可搬性:** 同期を OS のセマフォに寄せるので x86 / ARM の差は吸収できるが、
  macOS での動作確認は手動になる。
- **規模の上限:** numpy のままでは 5000 台で約 2.2 ms、1 万台で約 4.6 ms。
  1 万台で 10x を超える目標には P6、50x には案 D の検討が必要になる。

## Prototype

現行コアの計測スクリプト (`steady_bench.py`、`static_bench.py`) は P0 で `benchmark/steady_bench.py` に
正式化したので、ここからは外した。

- `prototype/soa_kernel.py`: 同じ負荷の SoA + numpy カーネル
  (`python3 docs/design/2026-10-26-fleet-scaling/prototype/soa_kernel.py 1000`)。
  円近似の narrow phase を使うため、PyBullet のメッシュ判定とは厳密には一致しない。
  出力は step ごとの中央値 (numba 版と同じ統計量)。
- `prototype/soa_static.py`: 倉庫型シーン (200 台移動 + 約 33k 静的な箱) の SoA + numpy 版。静的ボディのセルキーは一度だけソートし、
  毎ステップは移動体だけをキー付けして `searchsorted` で候補を引く
  (`python3 docs/design/2026-10-26-fleet-scaling/prototype/soa_static.py 200 100`)。
- `prototype/numba_kernel.py`: 同じ負荷の Numba 版 (counting sort のグリッド + 並列の近傍探索)。
  Numba はリポジトリの依存に入っていないので、別の venv で実行する
  (`NUMBA_NUM_THREADS=1` で 1 スレッドの値を測れる)。候補ペア数が numpy 版と一致することを確認済み。
  グリッドは密 (セル数 = nx × ny) で、広いマップには向かない (P6 参照)。
