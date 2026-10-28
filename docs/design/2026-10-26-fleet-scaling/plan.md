# Fleet Scaling — Implementation Plan

**Date:** 2026-10-26 (2026-10-27 改訂: レビュー反映)
**Status:** Draft (spec.md の Scope Approval 後に確定)
**Spec:** [spec.md](spec.md)
**基準コミット:** upstream `6551125` (行番号付きリンクはすべてこのコミット)

**Goal:**
- コア単体: 1000 台・headless・衝突判定ありで RTF 50x 以上 (dt=0.1 で 2.0 ms/step 以下)。
  10000 台では RTF 5x 以上 (20 ms/step 以下)。
- ROS ブリッジ込み: 1000 台、FleetState を sim time 10 Hz で publish し、RTF 10x 以上。
- 基準ホスト: WSL2 (16 cores、rendering なし)。判定は 3 回計測の中央値。

**Approach:** 状態の正本を SoA 配列に移し、ステップ単位の numpy カーネルで処理する。
PyBullet は遅延同期のミラーにする。コストの大きいフェーズから順に置き換え、各 PR の後に
定常状態ベンチマークで効果を確認する。

**前提となる upstream PR:** #94 (broad phase) と #87 (PathRequest) を先にマージする。
P1 は #94 の意味論を、P3 は #87 の request 状態を、それぞれ前提にする (spec の「関連する upstream PR」)。

---

## 予算 (1000 台, 2.0 ms/step, 疎シナリオ)

現状は `6551125` の 2026-10-27 の計測 (2 回の範囲)。

| フェーズ | 現状 | 目標 | 根拠 |
|---|---|---|---|
| collision (broad + narrow) | 18.7-19.6 ms | ≤ 0.8 ms | プロトタイプ 0.6-1.1 ms (#94 マージ後の現状値は取り直す) |
| AABB / grid 更新 | 4.8-5.4 ms | ≤ 0.1 ms | AABB を pose から配列計算すれば、grid は broad phase 内で毎回作り直せる |
| advance + Agent 更新 | 5.4-6.0 ms | ≤ 0.5 ms | カーネル 0.1-0.2 ms + イベント処理 |
| pose flush (PyBullet) | 1.6-1.8 ms | ≈ 0 ms (headless) | 遅延ミラー |
| step の固定オーバーヘッド | 約 0.1 ms | ≤ 0.2 ms | `getConnectionInfo`、イベント emit など |
| **合計** | **30.5-32.8 ms** | **≤ 2.0 ms** | 余裕 0.4 ms。プロトタイプにない static / mode 混在 / margin の処理をここで吸収する |

予算に余裕がないのは advance + Agent 更新である。Python の `Agent.update()` は 1 回約 1.4 µs
なので、1000 台全員を毎ステップ呼ぶだけで 1.4 ms を使い切る。したがって P3 (イベント駆動化) は必須とする。

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
| step (コア) | 30.5-32.8 ms | ≤ 2 ms (上の予算) |
| スナップショット書き込み (SoA → 共有メモリ) | なし (現状は in-process で `get_states` 約 6 ms + 構築約 3 ms) | ≤ 0.2 ms |
| 合計 | 約 42 ms (2.4x 相当) | ≤ 2.5 ms |

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

- `prototype/steady_bench.py` を `benchmark/` の正式なシナリオとして追加する
  (`run_benchmark.py --type steady` など、既存 CLI に合わせる)。1000 台と 10000 台を計測対象にする。
  `batch_perf._make_manager_and_agents` に依存しているので、公開関数 (`make_fleet()` など) に
  改名してから使う。
- シナリオは 2 つ用意する。
  - **疎 (sparse):** 現行の steady_bench と同じ。2 m 間隔のグリッドを全台が +x に平行移動する。
    AABB の重なりがほぼないので、broad phase とカーネルの固定費を測る。
  - **密集・交差 (dense):** 2 群が対向してすれ違う (間隔 1 m、margin 0.05 m) のに加え、
    1 割のロボットを停止させて他がその周りを通る。候補ペア、narrow phase の呼び出し、
    `COLLISION_STARTED` / `COLLISION_ENDED` の数をステップあたりで記録する。
    P4 の効果とイベント処理のコストはこのシナリオで測る。
  - **搬送 (carrying):** 疎シナリオの全台がベースに 1 個ずつ物体をアタッチして移動する。倉庫での通常形
    (ロボットがビンを運ぶ) で、現行では子オブジェクトの pose 伝播 (`_propagate_to_attached`、
    [sim_object.py#L1412](https://github.com/yuokamoto/PyBulletFleet/blob/6551125ec55a8a2861f0f18a0ad810f20e4f1b81/pybullet_fleet/sim_object.py#L1412))
    と子の AABB/grid 更新で phase 1-3 がほぼ倍になる。
  - **待機 (idle):** 1000 台が全台停止。現行は moved-only の差分判定で約 0.37 ms/step
    (`ros2_bridge/PERFORMANCE.md` の "Core only")。P1 の配列 broad phase がこれより遅くならないことを見る。
- 計測規約: 基準ホストは WSL2、warmup 20 steps、計測 150 steps × 3 回、中央値を採用。
  出力は per-phase の mean と median の両方を出し、判定には 3 回の median を使う。
- `docs/benchmarking/results.md` に現状のベースライン (spec の計測表) を記録する。
  #94 がマージされたら、その直後のコミットで取り直す。
- 2D OBB (SAT) の narrow phase を `prototype/soa_kernel.py` に追加し、密集シナリオでの
  1 ペアあたりのコストを測って、上の予算表を確定する。
- **効果:** なし (計測の土台)。
- **検証:** 3 回計測のばらつきを記録する。

### P1. SoA state store と配列ベースの衝突 broad phase

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
  以降は pose から配列で AABB を計算する。多リンク URDF は従来どおり `getAABB` を使う。
- broad phase: セルキーのソートと `searchsorted` で候補ペアを作る
  (`prototype/soa_kernel.py` の方式)。登録範囲と探索は #94 と同じ
  (margin/2 で広げた AABB が届く全セルに登録し、自セルだけ探索、NORMAL_2D は z を広げない)。
  セルより大きいボディ (コンベア、エレベータの床、壁) は届くセルごとにキーを繰り返して登録する
  (`soa_kernel.py` は中心セル 1 つ + 3×3 探索なので、このままでは大きなボディのペアを落とす)。
  static の除外、`CollisionMode` (2D/3D、DISABLED) の扱いは配列のマスクで再現する。
  **moved-only:** グリッドのキーは持続させ、動いた行だけキーを更新して近傍探索する
  (動いた割合が閾値を超えたら全件ソートし直す)。全台停止のシナリオで現行の約 0.37 ms を下回ること。
- `query_aabb()` (#98) は store の AABB 配列に対する重なり判定で提供する。
  「最後の step 以降に pose が書かれたオブジェクト」は dirty マスクで、DISABLED は mode マスクで扱う。
- narrow phase は、この段階では従来どおり PyBullet (`getClosestPoints` / `getContactPoints`)。
- 衝突結果 (active pair の集合と、その差分) を object_id の配列で持ち、
  `stop_on_collision` (#96) など controller 側の消費もその配列で行う。
- `_cached_aabbs_dict`、`_cached_spatial_grid`、`_cached_object_to_cell` を削除する。
- **効果見込み:** collision と AABB/grid 更新で 24 ms → 2 ms 前後。
- **検証:** 衝突 parity テスト: スクリプト化したシーン (アタッチ/デタッチ、`CollisionMode` の変更、
  `set_collision_margin`、ignore filter、`physics: true` のボディを含む) で、#94 と
  `COLLISION_STARTED` / `COLLISION_ENDED` のイベント列 (ペアとステップ番号) と `signed_distance` が
  一致すること。「同じ結果」と「文書化した差」をテストで区別する。
  `query_aabb` parity テスト (`tests/test_query_aabb.py` がそのまま通ること)。
  `tests/` の collision 系テストがすべて通ること。
- **リスク:** 最大の PR になる。`SimObject` の pose 経路は two-phase step の契約
  (`docs/architecture/two-phase-step.md`) に関わるので、契約は変えずに内部の保存先だけを変える。

### P2. PyBullet 遅延ミラー

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
- **効果見込み:** pose flush 1.6-1.8 ms → headless ではほぼ 0。
- **検証:** GUI あり・なし、`mirror: always` あり・なしで同じ衝突結果になること。
  recorder の出力が変わらないこと。callback 中の `p.getBasePositionAndOrientation` が
  `mirror: always` で前ステップの値を返すこと (現行契約)。

### P3. batch controller の store 直結と Agent 更新のイベント駆動化

現状の phase1 (5.4-6.0 ms) の内訳 (`6551125`、1000 台 diff、batch): batch カーネル本体 0.27 ms、
`_apply_phase1` 3.85 ms (うち `set_poses` 3.2 ms、per-agent の速度コピーループ
[batch_base.py#L474](https://github.com/yuokamoto/PyBulletFleet/blob/6551125ec55a8a2861f0f18a0ad810f20e4f1b81/pybullet_fleet/controllers/batch_base.py#L474)
約 0.6 ms)、アクションを持たない 1000 台の `Agent.update()` ループ約 0.55 ms。
カーネルは問題でなく、周りのループが問題である。

- batch controller のバッファ (`_pos_buf`, `_orn_buf`, `_lin_vel_buf`, `_ang_vel_buf`) を store の行に
  置き換え、`set_poses` 経由の書き戻しと速度のコピーループをなくす。`Agent.velocity` /
  `_current_velocity` は store の行のビューにする。
- `step_once()` の per-object ループを、「Python の更新が必要なオブジェクト」の集合だけに回す。
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
- **効果見込み:** 5.4-6.0 ms → 0.5 ms 以下 (カーネル 0.27 ms + イベント処理)。
  `MoveAction` 経由で全台を動かす構成でも同じ値になること。
- **検証:** アクション、BT、プラグイン、ROS ブリッジのテスト。batch と per-agent で同じ軌跡になること。
  疎シナリオを `MoveAction` 経由 (BT と同じ経路) で動かした場合も予算内であること。
  battery プラグインありの 1000 台で、`on_batch_update` の有無で `battery_soc` が一致すること。

### P4. プリミティブ narrow phase (箱と円を既定にする)

- 円・箱 (2D OBB) 同士の距離判定を numpy で実装する。`collision_margin` と signed distance を扱う。
  判定は「z 区間の重なり (NORMAL_2D では margin なし、現行の broad phase と同じ) AND
  xy 平面のプリミティブ距離 ≤ margin」とする。
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

- fleet services (`navigate`、`stop`、`attach`、`joint_command`) と per-robot の goal 系のハンドラを、
  「検証してキューに積み、`accepted` の ack を返す」形にする。
  sim スレッドは `step_once()` の先頭 (PRE_STEP の前) でキューをまとめて適用する。
- 適用時の拒否 (`set_goal_pose()` が `False` など) は、コマンド ID 付きのイベント (`/fleet/command_events` など) で通知する。
- `ack: applied` オプション (適用まで `Event` で待つ) を残し、RMF アダプタなど ack の結果で分岐するクライアントは
  これを使う。既定は `accepted`。
- sim の状態を直接変える services (`set_entity_state`、`reset_simulation`、`spawn_entity`、`delete_entity`) も
  同じキューを通し、別スレッドから step の途中で状態を書き換えないようにする。
- **効果見込み:** 最大速度で 1000 台の `/fleet/navigate` の ack が 0.55 s → 数 ms (ハンドラの中身が検証だけになる)。
  hybrid モードの 8 s はエンドポイント数が原因なので、この PR では直らない (使わない構成として文書化済み)。
- **検証:** `ros2_bridge/PERFORMANCE.md` の fleet モードの手順で、1000 台・最大速度の ack を before/after で記録。
  `docker/test_bridge_api.sh` の bridge 統合スモークテスト。RMF デモ (`ros2_bridge/pybullet_fleet_rmf`) が
  `ack: applied` で従来どおり動くこと。
- **判断:** この PR の後も最大速度で ack が 10 ms を超えるなら、コマンドリング (下記 follow-up) に着手する。

### P5. ROS ブリッジの状態配信を共有メモリ経由の別プロセスにする

設計は spec の「ROS ブリッジ: 状態配信の分離 (Q4)」を参照。2 つの PR に分ける。
分ける理由は、P5a が ROS に依存しない core 側の変更で、P5b が ROS 側の変更だからである。

**P5a. 共有メモリのスナップショット (core 側、ROS 非依存)**
- `multiprocessing.shared_memory` 上のリングバッファ (K スロット、既定 8) に、毎ステップ SoA の
  スナップショット (`seq`、`sim_time`、`generation`、`object_id` 列、状態列) を書く publisher を core に追加する。
  レイアウト (列名、dtype、ヘッダ) をバージョン付きで定義する。spawn / delete / reset
  (`/sim/spawn_entity` など) で行の集合が変わるたびに `generation` を上げ、名前と object_id の対応表
  ブロックを書き直す。reader は `generation` が変わったら対応表を読み直す。
- 同期は spec の「共有メモリの同期」のとおり: ヘッダの `head` / reader ごとの `tail`、
  スロットの引き渡しは OS のセマフォ (メモリバリアを兼ねる)、スロットごとの seqlock は整合性確認用。
- 配信モード `every_step` (既定、満杯なら sim が待つ。`wait_timeout` 既定 1.0 s、
  超過時は WARNING を出して `latest` 相当で進む) と `latest` (待たない) を実装する。
  reader の heartbeat と、reader 不在時に待たないことを含む。
- reader 側の `resource_tracker` 対策 (3.10-3.12 は `unregister()`、3.13 以降は `track=False`)。
- Python の reader API も用意する。ROS を使わない Python のツールやビューアも、同じ reader で状態を読めるようにする。
- **効果見込み:** sim 側の書き込みコストは 1000 台で数十 µs、1 万台で約 0.2 ms。
- **検証:** reader が破損したスナップショットを読まないこと (書き込み中の読み取りを含むテスト)。
  `every_step` で欠番がないこと。`latest` で sim が待たないこと。reader を kill しても sim が
  `wait_timeout` 以内に進むこと。reader 不在で sim が待たないこと。
  macOS (Apple Silicon) で整合性テストを手動で 1 回通す。

**P5b. ブリッジプロセス (ros2_bridge 側)**
- **着手条件:** `FleetState` は `RobotState3D[] robots` なので、rclpy では robot ごとの
  メッセージオブジェクト生成を避けられない (現状の構築 2.8 ms が下限)。P5b の前に
  「1000 台の FleetState を rclpy で構築して publish するだけ」の時間を単体で測り、
  4 ms 以内ならブリッジは rclpy、超えるなら最初から C++ にする。
- 共有メモリを読み、FleetState、`/clock`、TF、odom を publish する別ノード (別プロセス) を追加する。
  FleetState の型は変えない。必要なら、配列をそのまま載せる平坦なトピック
  (`FleetStateArray` など) を FleetState と並行して追加する (任意、既存の購読者には影響しない)。
- 現行の BridgeNode からは状態 publish を外し、services / actions だけを残す。
- `publish_rate` (sim time 基準) は両モードに適用する。
- **検証:** 受け入れ条件 6 の手順 (spec): 疎シナリオと同じ 1000 台全台移動・衝突判定毎ステップ、
  `every_step`、`publish_rate: 10`、同一ホストに FleetState の購読者 1 つ以上、`/clock` から
  60 s wall 以上で RTF を求め、3 回の中央値で 10x 以上。`ros2_bridge/PERFORMANCE.md` の現行手順は
  静止ワークロード・`publish_rate 5` なので、そのままでは使わない (手順を更新する)。
  ブリッジ側の 1 publish あたりの時間を記録し、上の予算表と比べる。
  `latest` モードで、ブリッジを起動しても sim のコアの step 時間が
  ROS なしの 1.05 倍以内であることを確認する。`docker/test_bridge_api.sh` の bridge 統合スモークテスト。
- **ブリッジを C++ にする判断:** rclpy で `every_step` の 1 回あたり publish コストが 8 ms に届かない場合。
  sim 側は共有メモリのレイアウトにしか依存しないので、P5a はそのまま使える。
- **範囲外 (follow-up):** 1 万台の全台 FleetState を毎ステップ配信すること。manager 単位のストリームに分け、
  複数のブリッジプロセスで publish する構成を検討する。
- **条件付き follow-up: コマンドリング。** PC の後も最大速度で `/fleet/navigate` の ack が 10 ms を超える場合に、
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

- **#94 マージ後 (P0 の途中):** ベースラインを取り直す。P1 の目標値は変えない。
- **P0 の後:** 密集シナリオの OBB プロトタイプの数値で、narrow phase の予算 (≤ 1.5 ms) を確定する。
  届かなければ、円近似の適用範囲を広げる (箱の縦横比が 1.5 未満なら円) か、P6 を前倒しする。
- **P1 の後:** 疎シナリオで collision + AABB が 2 ms 前後に下がっているか確認する。
  下がっていなければ、Python 側の固定オーバーヘッドを再プロファイルし、P2 以降の順序を見直す。
- **P3 の後:** 1000 台で 2.0 ms/step、10000 台で 20 ms/step を満たしているか確認する。
  満たしていれば P4、P5 は機能要件 (ROS、密集シーン) のために進める。
  満たしていなければ、P6 を前倒しするか検討する。
- **PC の後:** 最大速度で 1000 台の `/fleet/navigate` の ack が 10 ms を超えていれば、コマンドリングを follow-up から本計画に入れる。
- **P5b の後:** ブリッジ側の 1 publish が 8 ms を超えていれば C++ 化を判断する。
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
  #98 の `query_aabb()` と #87 の `MoveAction` も同じ理由で先に入れる。
- **ベンチマークのぶれ:** WSL2 上で ±10% 以上ぶれる。判定は中央値と相対比で行う。
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

- `prototype/steady_bench.py`: 現行コアの定常状態ベンチマーク (疎シナリオ)
  (`python3 docs/design/2026-10-26-fleet-scaling/prototype/steady_bench.py 1000 diff 60`)。
  `benchmark/batch_perf.py` の private 関数に依存しているので、P0 で公開関数に置き換える。
- `prototype/soa_kernel.py`: 同じ負荷の SoA + numpy カーネル
  (`python3 docs/design/2026-10-26-fleet-scaling/prototype/soa_kernel.py 1000`)。
  円近似の narrow phase を使うため、PyBullet のメッシュ判定とは厳密には一致しない。
  出力は step ごとの中央値 (numba 版と同じ統計量)。
- `prototype/numba_kernel.py`: 同じ負荷の Numba 版 (counting sort のグリッド + 並列の近傍探索)。
  Numba はリポジトリの依存に入っていないので、別の venv で実行する
  (`NUMBA_NUM_THREADS=1` で 1 スレッドの値を測れる)。候補ペア数が numpy 版と一致することを確認済み。
  グリッドは密 (セル数 = nx × ny) で、広いマップには向かない (P6 参照)。
