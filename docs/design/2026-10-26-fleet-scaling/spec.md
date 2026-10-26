# Fleet Scaling Specification

**Date:** 2026-10-26
**Status:** Draft (Q1-Q3 は 2026-10-26 に決定。Scope Approval 待ち)

## Context

PyBulletFleet は kinematics-first の大規模フリートシミュレータである。現状の定常状態
(1000台が全台移動中、dt=0.1 s、衝突判定は毎ステップ、headless、ROS なし) の計測値は
約 36-40 ms/step で、RTF は約 2.5-2.9x にとどまる (`prototype/steady_bench.py`)。

既存の `benchmark/batch_perf.py` は、ロボットがゴールに着くと停止する構成のため、
p50 は停止中、mean は移動中の区間を反映しており、定常状態の性能を示していない。

### 計測結果 (2026-10-26, WSL2, 16 cores, 1000 台, diff, batch controller)

| フェーズ | mean | 主な処理 |
|---|---|---|
| collision_check | 18-19 ms | `filter_aabb_pairs` の Python による AABB 重なり判定ループ |
| phase1_update | 8 ms | `set_poses` の per-object ループ、`_apply_phase1` の per-agent ループ、全 Agent の `update()` |
| phase3_aabb_grid_flush | 5 ms | per-object の `p.getAABB` と dict/set による空間グリッド更新 (グリッド更新が二重に走っている) |
| phase2_pose_flush | 2 ms | per-object の `p.resetBasePositionAndOrientation` |
| total | 35-40 ms | |

cProfile では、PyBullet の C API 呼び出し自体は安い (`resetBasePositionAndOrientation` は
約 1.6 µs/回、`getAABB` は約 0.9 µs/回)。遅いのは、それらを取り囲む Python 側の
オブジェクト単位の処理 (dict 参照、set 操作、Pose 生成、メソッド呼び出し) である。

### プロトタイプ (`prototype/soa_kernel.py`)

同じ負荷を SoA 配列 + 純 numpy (ステップあたり数回の配列演算、C++ なし) で処理した結果:

| N | advance | quat | broad + narrow (circle) | total | RTF (dt=0.1) |
|---|---|---|---|---|---|
| 1000 | 0.07-0.18 ms | 0.03 ms | 0.6-1.1 ms | 0.7-1.3 ms | 77-148x |
| 2000 | 0.10 ms | 0.05 ms | 1.0 ms | 1.2 ms | 86x |
| 5000 | 0.19 ms | 0.08 ms | 2.0 ms | 2.2 ms | 45x |
| 10000 | 0.33 ms | 0.15 ms | 3.9 ms | 4.4 ms | 23x |

結論: 1000台・50倍速の目標に対して、ボトルネックは PyBullet でも Python という
言語でもなく、**状態を Python オブジェクト単位で持ち、オブジェクト単位で処理している
データ構造**である。データ配置を SoA に変えるだけで、numpy のままでも目標に届く見込みがある。

## Problem

1. 状態 (pose, velocity, AABB, grid cell) が `SimObject` / `Agent` の属性と core の
   dict/set に分散しており、どの処理もオブジェクト単位の Python ループになる。
2. 衝突判定の broad phase がアルゴリズムとしては正しい (空間ハッシュ) が、
   Python の dict/set とタプルで実装されている。
3. PyBullet へのミラー (pose flush, getAABB) が、GUI や PyBullet narrow phase を
   必要としない headless 実行でも全オブジェクトに対して毎ステップ走る。
4. batch controller 管理下のエージェントも `Agent.update()` が毎ステップ呼ばれる。
5. ROS ブリッジの publish 間引きは sim time 基準 (`bridge_node.py:527`) のため、
   50倍速では wall time あたりの FleetState 生成回数が 50 倍になる。

## Goal

1. **コア単体:** 1000 台の移動ロボットを、headless かつ衝突判定ありで、RTF 50x 以上
   (dt=0.1 で 2.0 ms/step 以下) で動かす。
2. **ROS ブリッジ込み:** 1000 台で RTF 10x 以上、FleetState を sim time で 10 Hz
   (dt=0.1 では毎ステップ、wall time では 100 回/s) で publish する。
   それ以上の倍速が必要な場合は、publish レートではなく倍速の側を下げて運用する。
3. **規模の参考値:** 10000 台で RTF 5x (20 ms/step) をコア単体で満たす。
   1000 台 / 50x と同じ「台数 × 倍速」にあたる。

ROS 2 / Open-RMF ブリッジと URDF アーム + IK は、引き続き同じ公開 API で使えること。

## Decision (推奨案)

**案 B: PyBulletFleet 内に SoA ベースの kinematic core を作り、PyBullet を「必要なときだけ同期するミラー」にする。**
カーネルは numpy で実装し、関数シグネチャを配列入出力に固定することで、
将来 Numba / C++ (nanobind) / Rust (PyO3) に差し替えられるようにする。

### 比較した選択肢

| 案 | 内容 | 1000台/50x の見込み | 評価 |
|---|---|---|---|
| A | 既存構造のまま局所最適化 (二重更新の除去、dict 参照の削減など) | 2-3倍程度。届かない | 却下。B の P1 で一部を吸収 |
| **B** | **SoA core + numpy カーネル + PyBullet 遅延ミラー + イベント駆動の Agent 更新** | **プロトタイプで 0.7-1.3 ms。届く** | **採用** |
| C | B + ネイティブカーネル (Numba → C++/Rust) | 5000-10000 台で効く | B のカーネル境界が固まった後の条件付き拡張 |
| D | フルスクラッチの新シミュレータ (Rust/C++ core + Python binding) | 届くが過剰 | 現目標では却下。下記参照 |
| E | GPU (JAX / Warp / MJX) | 単一ワールド・イベント駆動・N=1000 では不利 | 却下 |

### フルスクラッチ (案 D) を今回採らない理由

- 目標 (1000台/50x) は、データ配置の変更だけで届くことがプロトタイプで示された。
  言語を変える必要はまだない。
- 維持必須の URDF アーム + IK は、PyBullet の URDF ローダ、FK (`getLinkState`)、
  `calculateInverseKinematics` に依存している。新コアではこれらを再実装するか、
  別ライブラリ (pinocchio など) に置き換える必要があり、目標と無関係な作業が大きい。
- ROS/RMF ブリッジは `Agent` / `SimObject` / `FleetStateProvider` などの公開 API に
  依存している (`ros2_bridge/` の直接 PyBullet 呼び出しは `sim_services.py:279` の 1 か所のみ)。
  新コアでも、この API の互換層が必要になる。
- 案 B の SoA core は、案 D の core と同じ形 (状態配列 + ステップ単位のカーネル) になる。
  B を進めれば、D に移るときにもそのまま core として切り出せるので、B は D への遠回りにならない。

**D を再検討する条件:** 目標が「1 万台 / 50x」のように、1 台あたりの予算が約 200 ns を下回る規模に
上がった場合 (下記 Reference 参照)。または、B の完了後に Python 側の固定オーバーヘッド
(イベント処理、ブリッジ) が予算を超えると計測で分かった場合。
外部事例では、ARGoS (C++、空間分割したマルチエンジン、4コアで 4000 台をリアルタイム) が
最も近い前例である。

## Architecture (案 B)

```
Python orchestration (変わらない公開 API)
  Agent / SimObject  ── row handle ──┐
  Actions, BT, plugins, ROS bridge   │ イベント (到着、衝突開始/解消) だけを受け取る
                                     ▼
SoA state store (core 所有の連続配列)
  pos(N,3) orn(N,4) vel(N,3) yaw(N) aabb(N,6) footprint(N) flags(N)
                                     │ ステップごとに 1 回ずつ呼ぶカーネル
  advance(dt) → aabb_from_pose() → broadphase() → narrowphase_primitive()
                                     │ 必要な行だけ
                                     ▼
PyBullet mirror (遅延同期)
  GUI 表示中 / メッシュや多リンクの narrow phase / アーム FK・IK / body_id に直接触れる API
```

### 設計上の要点

1. **状態の正本を SoA に置く。** `SimObject._cached_pose` などは、store の行を読み書きする
   アクセサに置き換える。`get_pose()` は従来どおり `Pose` を返す。
2. **カーネルはステップ単位で配列を受け渡す。** オブジェクト単位でカーネルを呼ばない。
   カーネル境界はそのまま、将来のネイティブ化の差し替え点にする。
3. **AABB は pose とローカル形状から計算する。** 単一ボディは store 内で計算し、
   多リンク URDF (アーム) は従来どおり PyBullet の `getAABB` を使う。
4. **narrow phase は 2 段にする。** 既定ではプリミティブ (箱または円) 同士を numpy で判定する。
   cylinder / sphere の形状は円、それ以外 (box、mesh、単一ボディの URDF) はローカル AABB から
   求めた箱 (2D OBB) で近似する。物理オブジェクト、多リンク URDF (アーム)、および
   `collision_shape: pybullet` を明示したオブジェクトを含むペアだけを PyBullet
   `getClosestPoints` / `getContactPoints` に回す。physics を使わない時点で形状の詳細は
   シミュレーションの目的に寄与しないため、近似を既定とする。
5. **PyBullet ミラーは dirty フラグで遅延同期する。** headless でミラーを必要としない
   オブジェクトには、毎ステップの `resetBasePositionAndOrientation` を呼ばない。
   `body_id` や `sim_core.client` を使う呼び出しの前には同期を保証する。
6. **Agent の Python 処理をイベント駆動にする。** `Agent.update()` を呼ぶのは、
   未完了の非移動アクション、関節補間、アタッチ、step ごとのプラグインがある Agent だけにする。
   batch カーネルは「到着した行」などのイベントだけを Python に返す。
7. **ブリッジの状態配信を、共有メモリ経由の別プロセスにする** (Decisions の Q4)。
   sim は毎ステップ SoA のスナップショットを共有メモリに書くだけにし、ROS メッセージの構築と
   publish はブリッジプロセスが行う。コマンド (services / actions) は当面 sim プロセス内に残す。

## Requirements

- 定常状態ベンチマークを `benchmark/` に正式に追加し、CI とローカルの両方で再現できること。
- 1000 台 diff、dt=0.1、衝突判定は毎ステップ、全台移動中、headless、ROS なしで、
  mean step 2.0 ms 以下 (RTF 50x 以上)。基準ホストと計測回数は plan で固定する。
- 既存のテストスイート (`make verify`) が通ること。
- 衝突判定の結果が一致すること (parity): `collision_shape: pybullet` の設定では、
  現行と同じ衝突ペアを返す。既定のプリミティブ近似では、テストシーンでの差を文書化する。
- 1000 台、ROS ブリッジあり、FleetState を sim time 10 Hz で publish する設定で、RTF 10x 以上。
- URDF アーム + IK の既存デモとテストが、挙動を変えずに通ること。
- ROS 2 / Open-RMF ブリッジの公開インターフェース (FleetState, navigate) は互換を保つこと。

## Constraints

- PyBullet は依存として残す (URDF、IK、GUI、物理モード、メッシュ衝突)。
- Python ≥ 3.10。新しいネイティブ依存 (C++ 拡張、Numba) は、この仕様の範囲では入れない。
- 物理モード (`physics: true`) は壊さない。ただし高速化の対象外とする。
- 後方互換は必須ではない (`.github/copilot-instructions.md`)。ただし ROS ブリッジと
  アームの公開 API は維持する。

## Non-goals

- 1 万台で 10x を超える倍速、マルチプロセス化、空間分割による並列化。
- ネイティブカーネル (案 C) の実装。カーネル境界の設計だけを範囲に含める。
- フルスクラッチの新シミュレータ (案 D)。
- 物理モードの高速化、GPU 化。
- GUI の別プロセス化 (Rerun / foxglove など)。必要になれば follow-up とする。

## Acceptance Criteria

計測はすべて基準ホスト (WSL2、16 cores、rendering なし) で行い、値は 3 回計測の中央値とする。

1. `benchmark/` の定常状態ベンチマークで、1000 台の mean が 2.0 ms/step 以下。
2. 同じベンチマークで、10000 台の mean が 20 ms/step 以下。2000 台と 5000 台は数値を記録する。
3. `make verify` が通る。衝突 parity テストと、アーム + IK のテストが通る。
4. ROS ブリッジのベンチマーク (`ros2_bridge/PERFORMANCE.md` の手順) で、1000 台、
   FleetState を sim time 10 Hz で publish し、RTF 10x 以上。

## Decisions (2026-10-26)

- **Q1. ブリッジ込みの目標は RTF 10x、FleetState は 10 Hz。** 10 Hz は sim time 基準とし、
  dt=0.1 では毎ステップの publish になる。wall time に換算すると 1 回あたりの予算は 10 ms で、
  その中に step、FleetState の生成、ROS メッセージへの変換、publish が収まる必要がある
  (現状は 1000 台で FleetState 生成だけで約 6 ms、ROS メッセージ構築が約 2.8 ms)。
  10x を超える倍速が必要な場合は、publish レートは保ち、倍速の側を下げる。
- **Q2. 衝突判定の既定は箱と円のプリミティブ近似。** physics を使わない時点で形状の詳細は不要で、
  メッシュ判定を既定にするとベンチマークの条件管理も難しくなる。PyBullet による判定は opt-in。
- **Q3. 基準ホストは当面 WSL2。** rendering なしなら WSL2 でも十分な速度が出る。
  ばらつきに対しては、3 回計測の中央値で判定する。
- **Q4. ROS ブリッジの状態配信は共有メモリ経由の別プロセスにする。** 既定では全ステップを配信し、
  間引くオプションを持つ。詳細は次節。

## ROS ブリッジ: 状態配信の分離 (Q4)

### 目的

ROS の有無で sim 本体の速度が変わらないようにする。ROS のコストは「publish 回数 × 台数」に
比例し、sim の step より重い (1000 台で 1 回あたり約 9 ms + DDS)。これを sim のスレッドから外す。

### 構成

```
sim プロセス                                   ブリッジプロセス (rclpy、将来は C++ も可)
  step_once()                                    共有メモリから最新、または次のスナップショットを読む
    └ 毎ステップ: SoA スナップショットを    ──▶     FleetState / /clock / TF / odom を構築して publish
      共有メモリのリングバッファに書く
      (1000 台で約 100 KB、1 万台で約 1 MB)
  BridgeNode (in-process, 当面そのまま)
    services / actions (navigate, attach, spawn ...) ◀── ROS クライアント
```

- **状態プレーン (高頻度・一方向):** sim → 共有メモリ → ブリッジプロセス。スナップショットには
  `seq`、`sim_time`、SoA 配列 (pose、velocity、moving、battery など FleetState に必要な列) を含める。
  読み書きの整合は、seqlock (書き込み前後で seq を進め、読み手は seq の一致を確認する) で取る。
- **コマンドプレーン (低頻度・双方向):** 現行の BridgeNode を sim プロセス内に残す。
  sim services 約 14 個、`simulate_steps` action、ロボット単位の actions/services、
  fleet services と数が多く、すべてを RPC 化するのは作業量に見合わない。
  コマンドの頻度は低く、sim の速度への影響は小さい。

### 配信モード

| モード | 動作 | 用途 |
|---|---|---|
| `every_step` (既定) | リングバッファが満杯なら sim が待つ。全ステップのスナップショットが必ず配信される | ROS 側で全状態が必要な場合。sim の速度はブリッジの publish 速度で頭打ちになる |
| `latest` | sim は待たない。ブリッジは最新のスナップショットだけを読み、途中のステップは捨てる | sim を最大速度で回し、ROS 側は間引いた状態で足りる場合 |

`publish_rate` (sim time 基準の間引き) は両モードに適用できるようにする。
`every_step` + `publish_rate: 10` は「sim time 10 Hz の状態を漏れなく配信する」設定になる。

### 注意点

- `every_step` で高倍速を出すには、ブリッジ側の 1 回あたりの publish コストを小さくする必要がある。
  1000 台・50x・毎ステップ配信では 500 回/s になり、1 回 2 ms 以下が必要になる。
  rclpy でメッセージを配列から一括構築して届かなければ、ブリッジプロセスを C++ にする。
  sim 側は共有メモリのレイアウトだけに依存するので、ブリッジの実装言語は後から変えられる。
- `/clock` はスナップショットの `sim_time` から publish する。状態と時刻が必ず同じステップを指すようにする。
- コマンドのレイテンシは sim time で見ると倍速に比例して大きくなる (現行と同じ性質)。

## Reference: 10000 台の場合

1000 台 / 50x と 10000 台 / 5x は、どちらも「台数 × 倍速」が 50000 で、1 台あたりの予算は同じ
約 2 µs/step である。ただし固定費 (ステップごとの Python オーバーヘッド) が台数で割られるぶん、
実際には 10000 台 / 5x のほうが楽になる。

### 計測 (2026-10-26, WSL2, 16 cores, 全台移動中, dt=0.1)

| 項目 | 1000 台 | 10000 台 |
|---|---|---|
| 現行コア: step | 37.5 ms (2.7x) | 426 ms (0.23x) |
| 現行コア: うち collision / AABB・grid / phase1 / pose flush | 20.2 / 5.8 / 9.2 / 2.2 ms | 231 / 65 / 99 / 31 ms |
| 現行: `FleetStateProvider.get_states()` | 5.9 ms | 75.7 ms |
| 現行: spawn (`spawn_agents_grid`) / RSS 増分 | 0.3 s / +49 MB | 4.2 s / +180 MB |
| SoA + numpy カーネル (`prototype/soa_kernel.py`) | 0.7-1.3 ms | 4.6 ms (22x) |
| SoA + Numba 1 スレッド (`prototype/numba_kernel.py`) | 0.037 ms | 0.29 ms |
| SoA + Numba 16 スレッド | 0.029 ms | 0.17 ms |

### 結論: 1 万台で必要になる内容

| シナリオ | step の予算 | カーネル | Python 側に必要なこと | 位置づけ |
|---|---|---|---|---|
| 10000 台 / 5x | 20 ms | numpy で足りる (4.6 ms) | P3 のイベント駆動化 (per-agent の Python 処理を O(イベント数) にする)。FleetState の一括生成 | 本計画 (P0-P5) の範囲で届く |
| 10000 台 / 10x | 10 ms | numpy でも届くが余裕は約 5 ms | 上記に加え、step の固定オーバーヘッドの削減 | P6 (Numba) を入れると安全 |
| 10000 台 / 50x | 2 ms | Numba / C++ が必須 (0.17-0.29 ms) | 1 台あたり約 150 ns しか残らない。ステップごとの Python の per-agent 処理はゼロにする必要がある | 案 D (ネイティブ core) の検討範囲 |

1 万台で新たに問題になる点:

- **カーネルは問題にならない。** ネイティブ化すれば 1 万台でも 0.3 ms 程度で、マルチプロセス化や
  空間分割は不要である。1 万台規模の律速は、カーネルではなく Python のオーケストレーションと ROS I/O になる。
- **ROS ブリッジ。** 全台の FleetState は 1000 台で 144 KB、1 万台で約 1.4 MB になる
  (`ros2_bridge/PERFORMANCE.md` から外挿)。10 Hz (sim time) × 5x では wall time で 50 回/s、
  約 72 MB/s のシリアライズになり、rclpy 1 プロセスでは現実的でない。Q4 の別プロセス構成で
  sim 本体からは切り離せるが、ブリッジ側では `latest` モードで間引くか、manager 単位のストリーム
  (既存の "Manager selective / distributed" 構成) に分割して複数のブリッジプロセスで publish する必要がある。
- **衝突イベントの数。** 密集シーンで衝突ペアが多いと、Python 側のイベント処理がペア数に比例する。
  イベントは配列でまとめて渡す API にする。
- **spawn と GUI。** spawn は 1 万台で約 4 s、RSS 増分は約 180 MB で許容範囲である。
  PyBullet GUI は 1 万ボディの表示に向かないので、1 万台で表示が必要なら、GUI を別プロセスの
  ビューア (Rerun など) にして、間引いたスナップショットを渡す構成を follow-up とする。
