# HugWBC — Documentación técnica del repositorio

Este repositorio implementa el paper **HugWBC** (Humanoid Whole-Body Control) sobre el robot **Unitree H1** en **Isaac Gym**, usando **PPO** con arquitectura de red adaptativa (teacher-student).

---

## 1. Estructura del repositorio

```
HugWBC/
├── legged_gym/
│   ├── envs/
│   │   ├── base/
│   │   │   ├── base_task.py           # Clase base de Isaac Gym (sim, viewer, env grid)
│   │   │   ├── legged_robot_config.py # Dataclass base de configuración (LeggedRobotCfg)
│   │   │   └── curriculum.py          # RewardThresholdCurriculum para command curriculum
│   │   ├── h1/
│   │   │   ├── h1.py                  # H1Robot (env principal)
│   │   │   ├── h1_config.py           # H1Cfg, H1CfgPPO
│   │   │   ├── h1interrupt.py         # H1InterruptRobot (variante con perturbaciones)
│   │   │   └── h1interrupt_config.py  # H1InterruptCfg, H1InterruptCfgPPO
│   │   ├── h1_2/
│   │   │   ├── __init__.py
│   │   │   ├── h1_2.py                # H1_2Robot (hereda H1Robot)
│   │   │   ├── h1_2_config.py         # H1_2Cfg, H1_2CfgPPO
│   │   │   ├── h1_2interrupt.py       # H1_2InterruptRobot (hereda H1InterruptRobot)
│   │   │   └── h1_2interrupt_config.py # H1_2InterruptCfg, H1_2InterruptCfgPPO
│   │   └── __init__.py                # Registro de tareas (task_registry)
│   ├── scripts/
│   │   ├── train.py                   # Punto de entrada para entrenamiento
│   │   └── play.py                    # Evaluación/visualización de política
│   ├── utils/
│   │   ├── task_registry.py           # Registro de envs + make_env/make_alg_runner
│   │   ├── terrain.py                 # Generación de terreno procedimental
│   │   ├── math.py                    # quat_apply_yaw, wrap_to_pi
│   │   └── isaacgym_utils.py          # get_euler_xyz en tensor
│   └── legged_utils/
│       └── observation_buffer.py      # Buffer circular para historial de obs (latencia)
├── rsl_rl/                            # Implementación PPO (fork de RSL-RL)
│   └── rsl_rl/
│       ├── algorithms/ppo.py          # PPO con simetría loss y WBC loss
│       ├── modules/                   # MlpAdaptModel (actor-critic adaptativo)
│       ├── runners/on_policy_runner.py
│       └── storage/
└── resources/
    └── robots/
        ├── h1/urdf/h1.urdf            # Robot H1 (19 DOF)
        └── h1_2/urdf/h1_2_handless.urdf # Robot H1-2 sin dedos (27 DOF)
```

---

## 2. Flujo de entrenamiento

```
train.py --task h1int
    └── task_registry.make_env("h1int")
            └── H1InterruptRobot(cfg, sim_params, ...)
                    └── H1Robot.__init__()
                            ├── _parse_cfg()
                            ├── BaseTask.__init__()  ← crea sim Isaac Gym
                            ├── _init_buffers()
                            ├── _prepare_reward_function()
                            └── _init_command_distribution()
    └── task_registry.make_alg_runner(env, "h1int")
            └── OnPolicyRunner(env, train_cfg)
                    └── PPO(MlpAdaptModel, ...)
```

**Loop de entrenamiento (cada iteración):**
1. `env.step(actions)` → simula `decimation=4` pasos de física por cada paso de control
2. `post_physics_step()` → termination check → compute reward → reset → compute obs
3. PPO recolecta rollouts → calcula ventajas → actualiza política

---

## 3. El robot H1

### 3.1 DOF y orden (19 joints)

El orden en Isaac Gym sigue la jerarquía del URDF (BFS desde el root):

| Índice | Joint | Grupo |
|--------|-------|-------|
| 0 | left_hip_yaw_joint | Pierna izq |
| 1 | left_hip_roll_joint | Pierna izq |
| 2 | left_hip_pitch_joint | Pierna izq |
| 3 | left_knee_joint | Pierna izq |
| 4 | left_ankle_joint | Pierna izq |
| 5 | right_hip_yaw_joint | Pierna der |
| 6 | right_hip_roll_joint | Pierna der |
| 7 | right_hip_pitch_joint | Pierna der |
| 8 | right_knee_joint | Pierna der |
| 9 | right_ankle_joint | Pierna der |
| 10 | torso_joint | Torso |
| 11 | left_shoulder_pitch_joint | Brazo izq |
| 12 | left_shoulder_roll_joint | Brazo izq |
| 13 | left_shoulder_yaw_joint | Brazo izq |
| 14 | left_elbow_joint | Brazo izq |
| 15 | right_shoulder_pitch_joint | Brazo der |
| 16 | right_shoulder_roll_joint | Brazo der |
| 17 | right_shoulder_yaw_joint | Brazo der |
| 18 | right_elbow_joint | Brazo der |

**IMPORTANTE:** El índice 3 = left_knee, índice 8 = right_knee. Hardcodeado en `_reward_joint_power_distribution` de H1Robot:
```python
shank_joint_index = torch.tensor([3, 8], ...)
```

El índice 11-18 son joints de brazo. Hardcodeado en `_reward_standing_joint_deviation`:
```python
penalize_joint_index = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18], ...)
```

### 3.2 Cuerpos rígidos del H1

- **Root body:** `torso_link` (el torso ES el link raíz en H1, no hay pelvis separada)
- **Pies (foot_name="ankle"):** `left_ankle_link`, `right_ankle_link`
- **Contactos penalizados:** elbow, torso, hip (×6), knee (×2) = **11 cuerpos**
- **Terminación:** contacto de `torso` con el suelo

### 3.3 Grupos de índices DOF calculados dinámicamente en `_create_envs`

```python
torso_inds    = joints con 'torso' en el nombre              → [10]
shoulder_inds = joints con 'shoulder_roll' o 'shoulder_yaw'  → [12,13,16,17]
elbow_inds    = joints con 'elbow' o 'shoulder_pitch'        → [11,14,15,18]
hip_inds      = joints con 'hip_roll' o 'hip_yaw'            → [1,5,6] aprox
```

Estos índices se usan en rewards de desviación del default.

---

## 4. Sistema de observaciones del H1

### 4.1 Dimensiones

```python
PROPRIOCEPTION_DIM = 63   # obs del propio cuerpo
CMD_DIM            = 10   # comandos al agente
CLOCK_INPUT        =  2   # señales de reloj para gait
PRIVILEGED_DIM     = 24   # obs con info del entorno (solo teacher)
TERRAIN_DIM        = 221  # mapa de altura (17×13 puntos)
# Total obs_buf = 63 + 10 + 2 + 24 + 221 = 320
# partial_obs   = 63 + 10 + 2 = 75 (lo que ve el student)
```

### 4.2 Composición del obs_buf (en orden de concatenación)

**Base (siempre visible = partial_obs):**
```
[0:3]        base_ang_vel × scale             (3)
[3:6]        projected_gravity                 (3)
[6:25]       dof_pos - default_dof_pos × scale (19)
[25:44]      dof_vel × scale                   (19)
[44:63]      actions (last)                    (19)
[63:73]      commands × commands_scale         (10)
[73:75]      clock_inputs (sin gait phases)    (2)
```

**Privileged (solo teacher, añadido con has_privileged_info=True):**
```
[75:78]      base_lin_vel × scale              (3)
[78:79]      jump_h_error (body_height cmd err)(1)
[79:81]      foot_clearance (2 pies)           (2)
[81:82]      friction_coeff normalizado        (1)
[82:88]      contact_forces pies (2×3)         (6)
[88:99]      collision_states (11 cuerpos)     (11)
```

**Terrain:**
```
[99:320]     measured_heights - base_height_target, clipped (221)
```

### 4.3 CMD_DIM = 10 (orden en commands tensor)

| Índice | Comando |
|--------|---------|
| 0 | lin_vel_x |
| 1 | lin_vel_y |
| 2 | ang_vel_yaw |
| 3 | gait_frequency |
| 4 | gait_phase (0=hopping, 0.5=walking) |
| 5 | gait_duration (siempre 0.5) |
| 6 | foot_swing_height |
| 7 | body_height offset |
| 8 | body_pitch |
| 9 | waist_roll |

### 4.4 Latencia de sensores

Si `randomize_control_latency=True`, los primeros `6 + num_dof*2` elementos (IMU + encoders) se retrasan mediante `ObservationBuffer`. La latencia es uniforme entre `[0, 0.02]` segundos.

### 4.5 Historial de observaciones

El student recibe los últimos `include_history_steps=5` pasos de `partial_obs` apilados en 3D `(batch, 5, 75)` a través de `ObservationBuffer`.

---

## 5. Sistema de comandos y gait

### 5.1 Gait clock

```python
gait_indices += dt * frequency  # mod 1.0
foot_indices[0] = (gait_index + phase) mod 1.0   # pie izquierdo
foot_indices[1] = (gait_index + 0)    mod 1.0    # pie derecho
clock_inputs[:, 0] = sin(2π × foot_idx[0])
clock_inputs[:, 1] = sin(2π × foot_idx[1])
```

- **Walking:** phase=0.5 → los pies están 180° desfasados
- **Hopping:** phase=0.0 → los pies están sincronizados

### 5.2 Curriculum de comandos

Ver §6.4 para el mecanismo completo. Resumen: grid 2D de (vel_x, yaw_vel) donde las celdas de alta velocidad se desbloquean cuando el tracking supera el umbral de éxito.

### 5.3 Modos de entornos

- **terrain_curriculum_mode:** El robot sigue un heading para navegar terreno
- **high_track_mode:** Entornos que sampean de la curricula de velocidades altas
- **standing_envs_mask:** ~10% de los envs reciben commands=0 para aprender a estar quieto

---

## 6. Sistema de recompensas

### 6.1 Rewards activos en H1 (escala no cero)

| Reward | Escala | Descripción |
|--------|--------|-------------|
| tracking_lin_vel | +2.0 | Tracking velocidad lineal XY |
| tracking_ang_vel | +3.0 | Tracking velocidad angular yaw |
| dof_pos_limits | -10.0 | Penaliza posiciones cerca del límite articular |
| lin_vel_z | -0.1 | Penaliza velocidad vertical del base |
| ang_vel_xy | -0.5 | Penaliza rotación roll/pitch del base |
| hip_deviation | -2.0 | Penaliza hip roll/yaw lejos del default |
| shoulder_deviation | -1.0 | Penaliza shoulder+elbow lejos del default |
| standing_joint_deviation | -2.0 | Penaliza joints de brazo lejos del default durante standing |
| joint_power_distribution | -0.5 | Penaliza distribución desigual de potencia en rodillas |
| no_fly | +0.25 | Premia contacto apropiado según gait |
| termination | -200 | Penaliza terminaciones no-timeout |
| dof_vel_limits | -2.0 | Penaliza velocidades articulares excesivas |
| action_rate | -0.01 | Penaliza cambios bruscos de acción (1er y 2do orden) |
| feet_contact_forces | -0.2 | Penaliza fuerzas de contacto excesivas en pies |
| feet_slip | -0.2 | Penaliza deslizamiento de pies en contacto |
| feet_stumble | -0.2 | Penaliza contacto lateral de pies |
| dof_acc | -2.5e-7 | Penaliza aceleración articular |
| torques | -5e-6 | Penaliza torques aplicados |
| orientation_control | -20.0 | Tracking del pitch del cuerpo (body_pitch cmd) |
| base_height | -40.0 | Tracking altura del base (body_height cmd) |
| stand_still | -5.0 | Penaliza ancho de postura cuando cmd=0 |
| hopping_symmetry | -5.0 | Penaliza asimetría de pies en hopping |
| standing_air | -1.0 | Penaliza si ambos pies en aire durante standing |
| alive | +0.2 | Reward constante por sobrevivir |

### 6.2 Curriculum de penalización (penalize_curriculum)

`curriculum_scale` multiplica los rewards de **penalización** listados en `reward_curriculum_list` (action_rate, torques, dof_acc, base_height, stand_still, orientation_control, etc.). Comienza en `curriculum_init=0.2` y crece con la fórmula:
```python
curriculum_scale = curriculum_scale ^ penalize_curriculum_sigma   # sigma=0.8
```
aplicada cada N iteraciones (`training_curriculum`, por defecto cada 100 iters en H1). A mayor intervalo, más despacio suben las penalidades.

**Cronograma según intervalo** (para llegar a ~0.95):
- Intervalo 100 iters → saturación a iter ~2.400
- Intervalo 200 iters → saturación a iter ~3.200  
- Intervalo 400 iters → saturación a iter ~6.500

El robot enfrenta penalidades completas a partir de ese iter. Si el intervalo es demasiado corto (100 iters para H1-2), las penalidades suben antes de que el robot aprenda a caminar → colapso. Si es muy largo (800 iters), las perturbaciones de brazo nunca maduran.

**Para H1-2 se usa intervalo 200** (override en `h1_2interrupt.py::training_curriculum`).

`reward_curriculum_list` incluye todos excepto `tracking_lin_vel` y `tracking_ang_vel` (esos siempre tienen escala completa).

### 6.4 Curricula de comandos de velocidad (max_command_x / max_command_yaw)

**Sistema separado** de los anteriores. Usa `RewardThresholdCurriculum` (curriculum.py). No es un límite dinámico sino un **grid 2D de celdas con pesos** — las celdas de alta velocidad empiezan con peso=0 (no se sampean) y se van desbloqueando.

**Estructura del grid** (config h1int):
```
x_vel:   12 bins de -0.6 a 2.0  → tamaño bin ≈ 0.217 m/s  (limit_vel_x)
yaw_vel: 10 bins de -1.0 a 1.0  → tamaño bin = 0.200 rad/s (limit_vel_yaw)
```
Inicio: `set_to(low=[-0.6,-0.6], high=[0.6,0.6])` → solo celdas del rango inicial tienen peso > 0.

**Cómo se desbloquean celdas (curriculum.py::update, local_range=0.55):**
Cuando un env tiene éxito en su celda (bin), esa celda Y TODAS las adyacentes dentro de ±0.55 en cada eje reciben `weight += 0.2`. Esto abre celdas que antes tenían peso=0.

**Por eso el crecimiento NO es lineal — avanza en saltos de ~0.55:**

Para x_vel (centros relevantes: 0.59, 0.81, 1.02, 1.46, 1.68, 1.89):
- Éxito en ≈0.59 → desbloquea hasta 1.14 → abre bins 0.81 y 1.02
- Éxito en ≈1.02 → desbloquea hasta 1.57 → abre bins 1.46 y 1.68
- Éxito en ≈1.46 → desbloquea hasta 2.01 → toca el límite 2.0

Para yaw_vel (centros: 0.5, 0.7, 0.9):
- Éxito en ≈0.50 → desbloquea hasta 1.05 → abre 0.7 y 0.9 → directamente al límite 1.0
- **Yaw sube de 0.6 a 1.0 de golpe** porque el salto de 0.55 alcanza el límite en un paso

**Los límites (x=2.0, yaw=1.0) son hard-coded** en `limit_vel_x` y `limit_vel_yaw` — son los bounds del grid, no hay bins más allá.

**Umbral de éxito:** `tracking_lin_vel > 0.80` (más exigente que el disturb que usa 0.60). El robot necesita seguir la velocidad actual con >80% de calidad para desbloquear velocidades mayores.

- Loggeado como `max_command_x` y `max_command_yaw` en Tensorboard
- En h1int sano: satura a ~2.0 m/s (x) y ~1.0 rad/s (yaw) en iter ~5.000
- Independiente de `curriculum_scale` y de `disturb_rad_curriculum`

---

## 7. Controlador PD y torques

```python
torques = p_gains * (action * action_scale + default_dof_pos - dof_pos + motor_offsets) 
          - d_gains * dof_vel
torques *= motor_strength
torques = clip(torques, -custom_torque_limits, custom_torque_limits)
```

- `action_scale = 0.25` (las acciones son deltas en radianes)
- `decimation = 4`: 4 pasos de física por cada paso de control
- `dt = decimation × sim_dt = 4 × 0.005 = 0.02s` → 50Hz de control
- Las ganancias PD se mapean por substring del nombre del joint

---

## 8. Domain Randomization

| Parámetro | Rango |
|-----------|-------|
| friction | [0.1, 2.75] |
| stiffness multiplier | [0.8, 1.2] |
| damping multiplier | [0.8, 1.2] |
| motor_strength | [0.8, 1.2] |
| control_latency | [0, 0.02] s |
| inertia ratio | [0.8, 1.2] |
| mass ratio | [0.8, 1.2] |
| link_com_offset | ±0.01 m |
| added base mass | [-3, 9] kg |
| motor_offset | [-0.02, 0.02] rad |
| push velocity XY | hasta 0.6 m/s (modulado por curriculum) |
| push angular vel | hasta 0.6 rad/s |

---

## 9. Terreno procedimental

- Tipo: `trimesh` con curriculum de dificultad
- Puntos medidos: grilla 17×13 = 221 puntos alrededor del robot
- `horizontal_scale = 0.05m`, `vertical_scale = 0.005m`
- Proporciones de terreno: [1.0] (solo un tipo en la config base)
- El nivel de terreno aumenta si el robot recorre >50% de la distancia objetivo
- El nivel baja si recorre <50%

### 9.1 Foot scan

Grilla 7×3 = 21 puntos alrededor de cada pie para medir clearance local.

---

## 10. Arquitectura de red (MlpAdaptModel)

Arquitectura teacher-student:

**Actor (student, en deploy):**
- Input: `(batch, history_steps=5, partial_obs=75)` → Transformer/MLP encoder → latent (32)
- MLP: latent → [256, 128, 32] → num_actions

**Critic (teacher, solo training):**
- Input: `full_obs = partial_obs + privileged + terrain` (318 dims)
- MLP: [512, 256, 128] → 1 (value)

**Privileged encoder:**
- Input: privileged_dim=24 + terrain_dim=221
- Entrena para reconstruir `privileged_recon_dim=3` primeros elementos (base_lin_vel)

### 10.1 PPO especial

- `use_wbc_sym_loss=True`: loss de simetría corporal (penaliza asimetría left/right)
- `symmetry_loss_coef=0.5`
- `sync_update=True`
- `entropy_coef=0.01`

---

## 11. Terminación de episodios

Un episodio termina si:
1. Contacto del `torso` (root body) con el suelo
2. `|roll| > 0.8 rad` O `|pitch| > 1.0 rad` (orientación excesiva)
3. `|projected_gravity XY| > 0.8` (caída severa)
4. Timeout: `episode_length_buf > max_episode_length`

Solo (1), (2), (3) activan el reward de terminación (-200). El timeout no.

---

## 12. Registro de tareas

```python
# legged_gym/envs/__init__.py
task_registry.register("h1int",   H1InterruptRobot,   H1InterruptCfg(),   H1InterruptCfgPPO())
task_registry.register("h1_2",    H1_2Robot,          H1_2Cfg(),          H1_2CfgPPO())
task_registry.register("h1_2int", H1_2InterruptRobot, H1_2InterruptCfg(), H1_2InterruptCfgPPO())
```

**H1** (robot base) no está registrado como tarea separada.

```bash
python legged_gym/scripts/train.py --task h1_2int   # tarea principal
python legged_gym/scripts/play.py  --task h1_2int --load_run <nombre_run>
python legged_gym/scripts/train_walk.py --task h1_2walk --algo [ppo|ars] --headless  # Caso 4
```
Logs en `logs/<experiment_name>/`.

---

## 13. H1InterruptRobot — tarea `h1int`

Hereda de H1Robot y simula intervenciones físicas en los brazos (el "hug" del paper HugWBC).

### 13.1 Cómo funciona la perturbación

- `disturb_dim = 8`: actúa sobre los **últimos 8 DOF**, que en H1 son los joints de brazo (índices 11–18)
- `replace_action = True`: cuando hay perturbación activa, **reemplaza** la acción del agente en esos joints con una posición objetivo de ruido
- `switch_prob = 0.005`: cada step hay 0.5% de probabilidad de activar/desactivar la perturbación
- `interrupt_in_cmd=True`: añade un flag binario al vector de comandos (índice 10, CMD_DIM pasa de 10 a 11)

### 13.2 Modos de perturbación

**Uniform noise** (`uniform_noise=True`, activo por defecto):
- Sampea posiciones objetivo dentro de los rangos articulares reales del H1
- **BUG en H1:** el código intenta zerear shoulder_yaw y elbow cuando el brazo está plegado, pero usa `targets[mask][:, [2,3]] = 0` que en PyTorch crea una copia y no modifica el tensor original. El zeroing nunca ocurre en producción para ninguno de los dos brazos.
- **BUG adicional en H1:** la condición de fold del brazo derecho es `targets[:, 5] > 0.5`, pero right_shoulder_roll se sampea en [−3.0, 0.3] — el máximo es 0.3, así que la condición nunca se cumple. El umbral simétrico correcto sería `> −0.5`.
- `h1_2int` corrige ambos bugs: usa `targets[mask, 2:7] = 0` (modifica in-place) y el umbral simétrico correcto para el brazo derecho.

**Gaussian**: sampea ruido gaussiano alrededor de la posición actual del joint.

### 13.3 Curriculum de perturbación

`disturb_rad_curriculum` es un valor **por entorno** (0→1.0) que escala la amplitud de las perturbaciones de brazo. Cuando = 0: incluso con `disturb_masks=True`, los brazos siguen la acción del policy sin perturbación efectiva. Cuando = 1.0: poses aleatorias a amplitud completa.

**Lógica de actualización** (en `update_disturb_curriculum_grid`, llamado en cada `_resample_commands`):
```python
# Solo para noise envs con disturb_masks=True
all_rew = command_sums["tracking_lin_vel"][env] / ep_len   # ep_len = min(1000, 500) = 500
success_threshold = 0.6 × reward_scales["tracking_lin_vel"]

# Equivale a: ¿el promedio de exp(-||vel_err||²/σ) superó 0.60?
if all_rew > success_threshold:     → +0.05 (clip a max_curriculum=1.0)
if all_rew < success_threshold / 2: → -0.05 (clip a 0)
else:                               → sin cambio
```

**Las dos condiciones necesarias para que suba:**

1. **Tracking de velocidad ≥ 60%**: `_reward_tracking_lin_vel()` (exponencial del error de velocidad XY) promedia > 0.60 en los últimos 10 segundos. Indica que el robot sigue bien la velocidad comandada.

2. **Episodios de al menos ~500 pasos**: El denominador de `all_rew` es siempre **500** (fijo, independiente de la duración real del episodio). Si el episodio terminó en 300 pasos, `command_sums` solo acumuló 300 steps pero se divide entre 500 → `all_rew` es 40% menor de lo esperado. El umbral se supera de forma regular solo cuando los episodios duran consistentemente >500 pasos (robot caminando estable).

**Consecuencia práctica:** el disturb solo empieza a crecer cuando el robot ya domina la locomoción básica. Con `penalize_curriculum` saturando en iter ~6.500 (intervalo 400) y ep_len estable >700 desde iter ~5.000, el disturb empieza a subir a iter ~6.500. Con intervalo 200, satura a iter ~3.200 y el disturb empieza a subir allí.

**Otros parámetros:**
- `disturb_rad = 0.2`: amplitud máxima de perturbación (en radianes, relativo al radio alrededor del punto medio)
- `noise_curriculum_ratio = 0.5`: 50% de entornos son "noise" (reciben perturbaciones); el otro 50% son terreno/velocidad
- La actualización se llama cada `resampling_time = 10s = 500 pasos` Y en cada reset de episodio

### 13.4 Mecanismo de fusión (disturb_curriculum_method=2)

```python
# noise_mean = lerp entre posición actual y acción del agente, ponderado por curriculum
noise_mean = curriculum * (dof_pos - default) + (1 - curriculum) * (action * scale)
# disturb_actions clipeadas en radio ±disturb_rad * curriculum alrededor de noise_mean
disturb_actions = clamp(disturb_actions, noise_mean - rad*curr, noise_mean + rad*curr)
```

Cuando curriculum=0: la perturbación no tiene efecto. Cuando curriculum=1: fuerza posiciones dentro de disturb_rad de la posición actual.

### 13.5 Terminación modificada

Cuando `disturb_masks[env] = True`, **no se termina el episodio** aunque el torso toque el suelo:
```python
self.reset_buf[self.disturb_masks] = False
```

**Condición de gravedad desactivada:** `H1InterruptRobot.check_termination` calcula `gravity_termination_buf` (`|projected_gravity_XY| > 0.8`) pero **no lo añade a `reset_buf`**. Las condiciones de reset efectivas en h1int (y h1_2int por herencia) son solo:
1. Contacto del cuerpo raíz con el suelo (excepto durante perturbación)
2. `|roll| > 0.8` o `|pitch| > 1.0`
3. Timeout

Esto es deliberado: durante perturbaciones externas, el tronco puede inclinarse transitoriamente sin que sea una caída real. Resetear por proyección de gravedad interrumpiría episodios donde el robot aún puede recuperarse.

### 13.6 Índices hardcodeados en H1InterruptRobot (brazo empieza en DOF 11)

```python
# h1interrupt.py
actions[:, 11:]        # _reward_action_rate_upper
actions[:, :11]        # _reward_action_rate_lower
out_of_limits[:, 11:] = 0  # _reward_dof_pos_limits
reward[:, 11:] = 0         # _reward_dof_acc
dof_vel[:, 11:]            # _reward_dof_vel_limits
```

### 13.7 Diferencias de rewards vs H1 base

```python
action_rate       = 0      # deshabilitado globalmente
action_rate_lower = -0.01  # cambios en piernas (DOF < arm_start)
action_rate_upper = -0.01  # cambios en brazos, suprimido durante perturbación
shoulder_deviation          # × ~interrupt_mask
collision                   # × ~interrupt_mask
standing_joint_deviation    # × ~interrupt_mask
base_height       = -40.0
stand_still       = -10.0
orientation_control = -10.0
standing_air      = -2.0
```

---

## 14. Implementación H1-2

> **✓ Verificado en Jun 2026** — corriendo en producción como `h1_2int`.

### 14.1 DOF del H1-2 (27 joints)

El URDF `h1_2_handless.urdf` tiene 27 joints revolute (sin dedos, con muñecas):

| Grupo | Joints | DOF |
|-------|--------|-----|
| Pierna izq | hip_yaw, hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll | 6 |
| Pierna der | ídem | 6 |
| Torso | torso_joint | 1 |
| Brazo izq | shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw | 7 |
| Brazo der | ídem | 7 |
| **Total** | | **27** |

Diferencias clave vs H1:
- Tobillo partido: `ankle_pitch` + `ankle_roll` (vs `ankle` simple en H1)
- 3 joints de muñeca por brazo: `wrist_roll`, `wrist_pitch`, `wrist_yaw`

### 14.2 Links del H1-2

- `pelvis` → root/base del robot (distinto de H1 donde el root es `torso_link`)
- `torso_link` → conectado al pelvis via `torso_joint` (revolute)
- `left_ankle_roll_link` / `right_ankle_roll_link` → los "pies"
- `left_wrist_yaw_link` / `right_wrist_yaw_link` → extremo de los brazos

### 14.3 Configuración H1_2Cfg

Cambios respecto a H1Cfg:

```python
PROPRIOCEPTION_DIM = 87   # 3+3+27+27+27 (vs 63 en H1)
num_actions = 27
pos = [0.0, 0.0, 1.05]   # altura inicial ligeramente mayor
base_name = "pelvis"      # root body (vs "torso" en H1)
foot_name = "ankle_roll"  # detecta left/right_ankle_roll_link
terminate_after_contacts_on = ["pelvis"]
penalize_contacts_on = ["elbow", "torso", "hip", "knee"]  # torso=torso_link
```

Control gains añadidos (ankle split y muñecas):
```python
stiffness:     ankle_pitch=40, ankle_roll=40, wrist_roll/pitch/yaw=10
damping:       ankle_pitch=2,  ankle_roll=2,  wrist_roll/pitch/yaw=0.3
torque_limits: ankle_pitch/roll=38, wrist=10
```

Los rewards son idénticos a H1 — no se añaden rewards nuevos para las muñecas.
Las muñecas quedan reguladas implícitamente por `torques`, `action_rate` y `dof_acc`.

### 14.4 H1_2Robot — overrides críticos

**`_reward_standing_joint_deviation`** — sobrescrita porque H1 usaba índices hardcodeados:
```python
# H1 (hardcoded):
penalize_joint_index = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18], ...)
# H1-2 (dinámico, incluye muñecas como parte del brazo completo):
arm_inds = shoulder_inds + elbow_inds + wrist_inds
```

**`_reward_joint_power_distribution`** — sobrescrita:
```python
# H1 (hardcoded): torch.tensor([3, 8], ...)
# H1-2 (dinámico):
knee_inds = [i for i, n in dof_names_lower if 'knee' in n]
```

**`_create_envs`** — extendida:
```python
self.wrist_inds = [i for i, n in enumerate(dof_names_lower) if 'wrist' in n]
self.knee_inds  = [i for i, n in enumerate(dof_names_lower) if 'knee'  in n]
```

### 14.5 Dimensiones de obs para H1-2

```
num_observations  = 87 + 10 + 2 + 24 + 221 = 344
num_partial_obs   = 87 + 10 + 2 = 99
critic_obs_dim    = 87 + 10 + 24 + 221 = 342
```

PRIVILEGED_DIM = 24 igual que H1: elbow×2 + torso×1 + hip×6 + knee×2 = 11 cuerpos.

### 14.6 H1_2InterruptRobot — tarea `h1_2int`

Archivos: `h1_2interrupt.py` y `h1_2interrupt_config.py`

Hereda de `H1InterruptRobot`. Toda la lógica de perturbación se hereda; solo se sobreescriben los métodos con índices hardcodeados de H1.

**Diferencias respecto a H1InterruptRobot:**

| Parámetro | H1 (`h1int`) | H1-2 (`h1_2int`) |
|-----------|-------------|------------------|
| `disturb_dim` | 8 | 14 |
| `arm_start_idx` | 11 (hardcodeado) | calculado dinámicamente |
| `noise_scale` | 8 valores | 14 valores (de URDF H1-2) |
| `CMD_DIM` | 11 | 11 (mismo) |
| `PROPRIOCEPTION_DIM` | 63 | 87 |

**arm_start_idx:** calculado en `_create_envs` como min índice de joints con 'shoulder', 'elbow' o 'wrist'. Para H1-2 debería ser 13 (12 DOF piernas + 1 torso).

**Métodos sobreescritos:**
- `_reward_action_rate_upper` / `_reward_action_rate_lower` — usan `arm_start_idx`
- `_reward_dof_pos_limits` — usa `arm_start_idx`
- `_reward_dof_acc` — usa `arm_start_idx`
- `_reward_dof_vel_limits` — usa `arm_start_idx`
- `_reward_joint_power_distribution` — usa `knee_inds` dinámico
- `_reward_standing_joint_deviation` — usa índices dinámicos + `~interrupt_mask`
- `Uniform_disturb_resample` — adapta condiciones para 14 joints
- `Gaussian_disturb_resample` — usa `self.disturb_dim` en vez de `8` hardcodeado

**Uniform_disturb_resample para H1-2:**
- Índice 1 = `left_shoulder_roll`. Si < 0.5 rad → arm folded → zeroed indices 2:7
- Índice 8 = `right_shoulder_roll`. Si > −0.5 rad → arm folded → zeroed indices 9:14

**Lógica del umbral de fold (por qué 0.5 y −0.5):**
El roll es simétrico especularmente: brazo izquierdo se abduce con valores positivos (rango −0.38~3.4), brazo derecho se abduce con valores negativos (rango −3.4~0.38). "Brazo plegado/cerca del cuerpo" = poca abducción:
- Izquierdo: roll < 0.5 rad (por debajo del umbral positivo → no abducido)
- Derecho: roll > −0.5 rad (por encima del umbral negativo → no abducido)

**H1 tenía dos bugs aquí:**
1. Umbral del brazo derecho `> 0.5` — el rango máximo es 0.34, nunca se cumplía, el brazo derecho no tenía protección de fold.
2. Asignación `targets[mask][:, [2,3]] = 0` devuelve copia en PyTorch; no modifica `targets`. El zeroing del brazo izquierdo tampoco ocurría.
H1_2InterruptRobot corrige ambos: usa `targets[mask, 2:7] = 0` (in-place) y el umbral `> −0.5`.

**noise_scale y noise_lowerbound** (14 valores) sacados de los límites del URDF:

| Joint | lower | upper | scale |
|-------|-------|-------|-------|
| l_shoulder_pitch | -3.14 | 1.57 | 4.71 |
| l_shoulder_roll | -0.38 | 3.40 | 3.78 |
| l_shoulder_yaw | -2.66 | 3.01 | 5.67 |
| l_elbow | -0.95 | 3.18 | 4.13 |
| l_wrist_roll | -3.01 | 2.75 | 5.76 |
| l_wrist_pitch | -0.4625 | 0.4625 | 0.925 |
| l_wrist_yaw | -1.27 | 1.27 | 2.54 |
| r_shoulder_pitch | -3.14 | 1.57 | 4.71 |
| r_shoulder_roll | -3.40 | 0.38 | 3.78 |
| r_shoulder_yaw | -3.01 | 2.66 | 5.67 |
| r_elbow | -0.95 | 3.18 | 4.13 |
| r_wrist_roll | -2.75 | 3.01 | 5.76 |
| r_wrist_pitch | -0.4625 | 0.4625 | 0.925 |
| r_wrist_yaw | -1.27 | 1.27 | 2.54 |

---

## 15. Parámetros críticos a no perder de vista

- `num_actions` debe coincidir exactamente con el nº de DOFs del URDF (sin fixed joints)
- `default_joint_angles` debe tener una key por cada joint name exacto del URDF
- Los keys de `stiffness/damping/torque_limits` se buscan por substring — deben ser únicos y no ambiguos
- `PROPRIOCEPTION_DIM` debe coincidir con la suma real de los primeros elementos en `_preprocess_obs`
- `PRIVILEGED_DIM` debe coincidir con la longitud del bloque privilegiado en `_preprocess_obs`
- `TERRAIN_DIM = len(measured_points_x) × len(measured_points_y) = 17×13 = 221`

---

## 16. Referencia de entrenamiento sano — h1int

Run de referencia: `Jun18_21-15-42_`, observado en iter ~22.000/40.000. GPU RTX 2000 Ada (16 GB).

### 16.1 Velocidad de convergencia de curriculas

| Métrica | Iter de saturación | Valor final |
|---|---|---|
| curriculum_scales | ~5.000 | 1.0 |
| max_command_x | ~5.000 | 2.0 m/s |
| disturb_curriculum | ~5.000 (0.93) | ~0.97 estable |

Si curriculum_scales no llega a 1.0 antes de iter 10k → problema.

### 16.2 Señales de episodio sano

| Métrica | Valor esperado a ~20k iters |
|---|---|
| mean_episode_length | ~970/1000 |
| rew_termination | −0.01 a −0.02 |
| rew_no_fly | 0.19–0.20 (de 0.25 máx) |
| rew_base_height | −0.17 a −0.19 |
| rew_orientation_control | −0.07 |
| rew_standing_air | −0.003 a −0.004 |

### 16.3 Tracking de velocidad

- `rew_tracking_lin_vel` baja de ~1.54 a ~1.31 cuando la curricula se activa (~5k) y luego sube a ~1.58. El dip temporal en 5k es normal.
- `rew_tracking_ang_vel` mismo patrón, ~1.66 a 22k.

### 16.4 Policy y losses

- **mean_noise_std**: cae de 1.0 → ~0.60 en los primeros 2.000 iters, luego se estabiliza en **0.62–0.65 el resto del entrenamiento**. No converge a 0.1 y eso es normal: la perturbación aleatoria tiene varianza intrínseca alta y no compensa ser más determinista. El std en play.py es irrelevante porque se usa la acción media.
- **sym_loss**: baja lento 0.0079 → 0.0053. Gradual pero constante — señal de que el aprendizaje no está muerto.
- **privileged_recon_loss**: 0.077–0.093, estable pasada la mitad del entrenamiento.
- **value_function loss**: 0.07–0.11, fluctúa en rango estable.
- **surrogate loss**: −0.003 a −0.005, pequeño y estable (PPO maduro).

### 16.5 Señales de alerta (entrenamiento que va mal)

- `mean_episode_length` que no llegue a ~900 antes de iter 15k
- `rew_termination` < −1.0 sostenido → el robot cae con frecuencia
- `disturb_curriculum` que no supere 0.5 antes de iter 10k
- `mean_noise_std` que NO baje de 1.0 después de 5k → la política no aprende nada
- `rew_tracking_lin_vel` que no se recupere del dip de 5k antes de iter 15k

### 16.6 Notas sobre play.py

- play.py fuerza `num_envs=1` automáticamente. Con ~9 GB libres en la GPU durante entrenamiento, no hay conflicto.
- Por defecto fuerza `standing_envs_mask=True` y `commands[:3]=0` → el robot está parado recibiendo perturbaciones en brazos a curriculum completo (1.0). Para ver caminar, comentar esas dos líneas.
- El terreno plano es una elección de play.py (`min_height=max_height=0.0`). Isaac Gym sí soporta terreno variado (trimesh); el entrenamiento lo usa. Para evaluación con terreno, subir `max_height` a 0.05–0.1.
- Bug histórico en play.py: `export_policy_as_jit` se importaba pero no existía en `legged_gym.utils`. Ya corregido eliminando ese import.

---

## 17. Notas de entrenamiento H1-2 (h1_2int)

### 17.1 Los tres sistemas de curricula y su interdependencia

Hay **tres curriculas independientes** que evolucionan en paralelo:

| Curricula | Variable loggeada | Umbral | Efecto |
|-----------|------------------|--------|--------|
| Penalización | `curriculum_scales` | N/A (fórmula periódica) | Escala fuerza de penalties |
| Velocidad de comando | `max_command_x`, `max_command_yaw` | tracking_lin_vel > **0.8** | Expande rango de velocidades comandadas |
| Perturbación de brazo | `disturb_curriculum` | tracking_lin_vel > **0.6** + ep_len > 500 | Intensifica perturbaciones de brazo |

Las curriculas de velocidad y perturbación dependen del tracking, que a su vez depende de cuánto presionan los penalties. Por eso las tres tienden a saturar juntas.

### 17.2 Intervalo del penalize_curriculum para H1-2

H1 usa intervalo 100 iters (en `h1.py::training_curriculum`). Para H1-2 (27 DOF, más complejo) esto causa colapso catastrófico: las penalidades suben antes de que el robot aprenda la locomoción básica.

Override en `h1_2interrupt.py::training_curriculum`:
```python
if self.cfg.rewards.penalize_curriculum and (self.learning_iter % 200 == 0):
    self.curriculum_scale = pow(self.curriculum_scale, ...)
```

**Lecciones aprendidas (runs h1_2int, Jun 2026):**
- Intervalo 100 → colapso a iter ~500-700. Penalties suben antes de que el robot aprenda a caminar.
- Intervalo 200 → funciona pero penalties saturan en iter ~3.200 y disturb sube tarde (solo 0.37 en iter 14k).
- Intervalo 400 → penalties saturan en iter ~6.500, pero disturb empieza a subir al mismo tiempo → ep_len baja al confluir ambas presiones.
- **Intervalo 150 → objetivo actual (ver §19).** Balance: penalties saturan en ~iter 4.000, dando tiempo suficiente para aprender locomoción antes de que disturb suba.

### 17.3 Patrón ep_len↓ + noise_std↑

Indica que el robot enfrenta penalidades completas y disturb creciente al mismo tiempo sin haber consolidado la política. La solución es siempre que `penalize_curriculum` sature **antes** de que `disturb_curriculum` empiece a subir significativamente — de ahí la importancia del intervalo correcto en §17.2.

### 17.4 GPU paralela (h1_2int, RTX 2000 Ada 16 GB)

- Un run con 4096 envs usa ~7.5 GB.
- Para correr dos runs en paralelo: segundo run con `--num_envs 2048` (~4 GB) → total ~11.5 GB, seguro.
- No hay conflicto programático entre runs (logs separados en `logs/h1_2_interrupt/<timestamp>/`).

### 17.5 Gains PD del tobillo: Kp=40 correcto para simulación a 50Hz

El URDF del H1-2 especifica `effort=60 Nm` para `ankle_pitch` (vs 40 Nm del H1). La documentación oficial de Unitree para PR Mode recomienda `Kp=80, Kd=1`. **Sin embargo, aplicar Kp=80 en simulación causa inestabilidad.**

**Razón:** Kp=80 está calibrado para el hardware real corriendo a ~500Hz. La simulación corre el control a **50Hz** (decimation=4). A menor frecuencia, el PD sobrecompensa y genera oscilaciones en el tobillo, degradando ep_len en ~30% durante el aprendizaje temprano.

**Config correcto para simulación:**
```python
stiffness:  ankle_pitch=40, ankle_roll=40   # igual que H1, estable a 50Hz
damping:    ankle_pitch=1,  ankle_roll=1    # reducido de 2 a 1 (recomendación Unitree)
torque_limits: ankle_pitch=60              # corregido al valor real del URDF (era 38)
```

Solo el `torque_limit` de ankle_pitch se actualiza al valor real (60 Nm vs 38 anterior).
El Kd=1 (vs 2 original) reduce el amortiguamiento sin causar inestabilidad.

### 17.6 Tobillo split H1-2 vs tobillo único H1

El H1 tiene un único `ankle_joint` (pitch). El H1-2 tiene:
- `ankle_pitch` (lower=-0.897, upper=0.524 rad, effort=60 Nm) — rango y capacidad similar al H1
- `ankle_roll` (lower=±0.261799 rad = ±15°, effort=40 Nm) — rango muy limitado

En hardware real, es un **mecanismo paralelo de 5 barras**: 4 joints (A, B paralelos + P pitch + R roll seriales virtuales). El SDK provee "PR Mode" que convierte comandos P/R a comandos A/B internamente. En simulación, los joints P y R se controlan directamente (simplificación válida para RL training).

El `ankle_roll` tiene autoridad limitada: con Kp=40 y ±15° → max torque P = 40×0.262 = **10.5 Nm**. El balance lateral recae principalmente en los joints de cadera (Kp=200).

### 17.7 Discrepancias URDF vs config (correcciones aplicadas)

| Joint | URDF effort | Config original | Config corregido |
|-------|-------------|-----------------|------------------|
| ankle_pitch | 60 Nm | 38 | **60** ✓ |
| elbow | 18 Nm | 35 | **18** ✓ |
| resto | — | ≤ URDF (conservador) | sin cambio |

El `elbow` tenía torque_limit=35 (casi el doble del URDF=18). La política podía aprender torques de codo que el hardware real no puede producir.

**Masa total URDF H1-2:** 66.98 kg (sin batería/cables). El real ~70 kg — diferencia cubierta por `added_mass_range=[-3, 9]` kg.

### 17.8 Oportunidad de mejora: manos como fixed links

Si el robot se desplegará con manos (ej. Dex3-1 de Unitree, 5 DOF/mano), añadirlas como **fixed links** en el URDF mejora la fidelidad de la dinámica del brazo:

- El `wrist_yaw_link` actual pesa solo 0.124 kg; la Dex3-1 pesa ~0.36 kg
- El COM de la mano está desplazado ~6 cm del wrist_yaw_joint → crea torque gravitacional real
- Simplemente añadir masa al link existente ignora el offset del COM
- Un `<joint type="fixed">` en Isaac Gym colapsa el link en el padre → **0 DOF extra**

Pendiente: obtener STL + masa + offset COM de la Dex3-1 para implementar.

---

## 18. Hallazgos de análisis cross-run (h1_2int, Jun 2026)

Obtenidos comparando todos los runs de h1_2_interrupt/ con TensorBoard EventAccumulator.

### 18.1 Libertad de brazos durante aprendizaje — el mecanismo real

El robot pasa por dos fases claramente distintas respecto al control de brazos:

**Fase 1 — Brazos libres (terrain_curriculum_mode=True):**
El robot controla sus brazos con la política. En h1interrupt.py línea 286:
```python
self.disturb_masks[:] *= self.noise_disturb_mode[:] * (~self.terrain_curriculum_mode[:])
```
Mientras `terrain_curriculum_mode=True`, `disturb_masks=False` → los brazos no son perturbados. La política los usa libremente para ayudarse en la locomoción y el balance.

**Fase 2 — Brazos controlados externamente (terrain_curriculum_mode=False):**
Cuando un env sale del terrain mode, `disturb_masks` puede activarse (prob 50%). Con `disturb_replace_action=True` (h1interrupt.py línea 358–363):
```python
cliped_actions[:, -disturb_dim:] = torch.where(disturb_masks, disturb_action_clip, cliped_actions[:, -disturb_dim:])
```
Los últimos `disturb_dim` DOF (brazos) son **reemplazados** por un target de perturbación externo. La política ya no controla los brazos — aprende a compensar con piernas y torso.

**Cronología por robot:**
- **H1int**: terrain avanza a iter ~1000 → Fase 2 empieza rápido. A iter 1000 disturb=0.001, a iter 2000 ya es 0.23. Fase 1 muy corta.
- **H1_2int (pre-fix)**: terrain NUNCA avanzaba (bug) → Fase 1 permanente. disturb=0 para siempre aunque el robot caminase bien.
- **H1_2int con sigmoid**: Fase 1 dura ~0–4000 iters, sigmoid libera envs gradualmente a Fase 2 desde iter ~3200.

### 18.2 Por qué la Fase 1 larga causa `rew_dof_vel_limits` muy negativo en H1-2

Durante Fase 1, la política mueve brazos y muñecas libremente en terreno difícil → velocidades articulares altas → `rew_dof_vel_limits` muy negativo (suma sobre 14 DOF de brazo, incluyendo 6 muñecas que pueden girar rápido).

El **patrón de tres fases** observado en todos los runs que sobreviven ≥ 6000 iters:

```
iter  0-3000: dov cae hasta -1.2  (aprendiendo con brazos libres en terreno difícil)
iter  4000:   dov sube a -0.60    (sigmoid libera envs a disturb mode → brazos restringidos)
iter  6000+:  dov sube a -0.15    (terreno avanza + disturb mode establece brazos)
```

Este patrón es **contraintuitivo**: el disturb que "agita" los brazos en realidad REDUCE las violaciones de velocidad, porque reemplaza acciones caóticas de la política por targets fijos del disturb. El run Jun21_19-02-25_ (sin sigmoid) mostró dov empeorando continuamente: -0.79 → -1.33 → -1.56 → -1.80 hasta que fue matado.

**Consecuencia para el criterio de terreno:** `terrains_level.dof_vel_limits = 0.1` (threshold = 0.1×-2.0×curr_scale ≈ -0.20) bloquea el avance de terreno durante la Fase 1. El criterio pasa naturalmente cuando dov vuelve a -0.15 a -0.20 (Fase 2 establecida, iter 6000–10000). El umbral es correcto — no necesita ajuste.

### 18.3 Secuencia exacta de colapso (observado en Jun21_10-04-01_, colapso a iter ~6500)

El orden de degradación de las señales es siempre el mismo:

```
iter    ep_len  no_fly  tracking  priv_rec  noise_std  disturb_c
3000:    822   0.171    1.363     0.083     0.717      0.00
4000:    809   0.161    1.335     0.081     0.785      0.43  ← CAUSA: disturb subió 0→0.43 de golpe
5000:    426   0.072    0.675     0.092     0.821      0.89  ← gait roto
6500:      4   0.001    0.009     0.435     0.843      0.96  ← colapso total
```

**Orden de las señales de alarma:**
1. `disturb_curriculum` sube demasiado rápido (step=0.05: de 0 a 0.43 en un check → inmediatamente peligroso)
2. `rew_no_fly` cae a la mitad (~1000 iters antes del colapso total — **alarma más temprana**)
3. `ep_len` se divide por 2
4. `noise_std` sube (política se vuelve más estocástica — señal tardía)
5. `privileged_recon_loss` salta de ~0.08 a 0.43 (confirmación de colapso)

Nota: después del colapso, `disturb_curriculum` baja solo (0.96 → 0.56 en 1800 iters). El robot cae en 4 pasos → tracking≈0 → `curr_is_down=True` → disturb decrece. El sistema es auto-corrector pero demasiado lento para evitar el colapso.

**Fix aplicado:** step=0.01 (vs 0.05 original) ralentiza el crecimiento ~5×, dando tiempo al robot para adaptarse.

### 18.4 Diferencias estructurales H1 vs H1-2 (métricas comparativas)

| Métrica | h1int @1k | h1_2_best @1k | h1_2_actual @1k | Interpretación |
|---------|-----------|---------------|-----------------|----------------|
| noise_std | **0.50** | 0.91 | 0.86 | H1 converge 2× más rápido a política determinista |
| sym_loss | 0.0079 | 0.0142 | 0.0131 | H1_2 consistentemente más asimétrico (27 DOF) |
| dov_vel_limits | -0.026 | -0.585 | -0.720 | H1_2 tiene 20× más violaciones de brazo al inicio |
| dov_vel_limits @10k | -0.151 | -0.183 | -0.189 | Convergen al mismo nivel a largo plazo |
| priv_recon @1k | 0.081 | 0.158 | 0.110 | H1_2 tarda más en estimar velocidad linear |
| priv_recon @3k | 0.103 | 0.096 | 0.090 | Se igualan a iter 3k |
| ep_len @1k | 910 | 792 | 678 | H1_2 tarda más en aprender a no caer |

**`noise_std` del run actual a iter 10k = 0.887** (vs h1_2_best = 0.751, h1int = 0.641). El disturb activo (0.48 a iter 10k) mantiene la política incierta — el robot no sabe qué postura de brazo vendrá. Es esperado con disturb activo y no indica problema por sí solo.

**`sym_loss` no predice colapso** — baja de 0.013 a 0.009 en todos los runs (buenos y malos) con el mismo ritmo. Inútil como alarma.

### 18.5 Tabla de utilidad de métricas para monitoreo

| Métrica | Útil para alarma | Por qué |
|---------|-----------------|---------|
| `rew_no_fly` | ✅ Alarma temprana | Cae ~1000 iters antes del colapso total |
| `disturb_curriculum` velocidad de subida | ✅ Causa directa | Si sube >0.2 en una sola actualización → problema |
| `privileged_recon_loss` | ✅ Confirmación | Salta de 0.08 a >0.4 cuando robot yace en suelo |
| `noise_std` subiendo | ✅ Señal tardía | Sube cuando política no puede reducir incertidumbre |
| `rew_dof_vel_limits` | ⚠️ Contexto-dependiente | Alto en Fase 1 (normal), bajo en Fase 2 (normal) |
| `terrain_random_uniform` | ✅ Fingerprint de config | ~1.5 = sin initial_disturb, ~5.25 = con initial_disturb |
| `sym_loss` | ❌ No predice nada | Evoluciona igual en runs buenos y colapsados |

### 18.6 Estado del run actual (Jun28___, iter ~16346)

```
ep_len:             764    ⚠️ Bajando desde pico 933 (iter 6k) — presión normal de disturb=0.83
disturb_curriculum: 0.833  ✅ Creciendo lento (+0.036/1k iters, desacelerando)
curriculum_scale:   1.000  ✅ Saturado desde iter ~4000
max_command_x:      1.78   ✅ Avanzando hacia 2.0
rew_no_fly:         0.145  ✅ Gait saludable (alarma en 0.09, margen 38%)
rew_termination:   -0.071  ✅ Caídas mínimas (alarma en -1.0, margen 14×)
rew_dof_vel_limits: -0.092 ✅ Fase 2 establecida
```

El declive de ep_len es proporcional al crecimiento de disturb (ver §18.1). Cuando disturb se estabilice (~0.90), ep_len debería recuperar parcialmente. Cronología: disturb llegaría a ~0.95 en iter ~22k-25k al ritmo actual.

---

## 19. Decisiones de configuración H1_2int vs H1int (Jun 2026)

Tabla de referencia de lo que está igual, diferente o añadido respecto a h1int:

| Parámetro | h1int | h1_2int | Decisión |
|---|---|---|---|
| `action_rate_upper` | -0.01 (8 DOFs brazo) | -0.01 (14 DOFs brazo) | Igual |
| `action_rate_wrist` | no existe (H1 no tiene muñecas) | **-0.25** (6 DOFs muñeca) | AÑADIDO — muñecas sin función locomotora, 25× más agresivo que upper. -0.05 era insuficiente (wrist_vel=30 rad/s sin bajar en iter 1301). Actualizado a -0.25 para el próximo run. |
| `wrist_deviation` | no existe | **-1.0** | AÑADIDO — señal de posición para clean envs en Phase 2 |
| `dof_vel_limits` (reward) | penaliza brazos (8 DOFs) | penaliza brazos **excluyendo wrists** (8 de 14 DOFs) | MODIFICADO — wrists oscilan a 30 rad/s como artefacto de target-chasing (no oscilación física); incluirlos bloqueaba terrain graduation. Exclusión correcta porque no tienen función locomotora |
| `terrains_level.dof_vel_limits` | 0.1 | **0.30** | Revertido al valor H1-2 original (0.30 fue el diseño; 0.1 fue workaround para bug de wrists que ya está resuelto vía exclusión) |
| `disturb.tracking_lin_vel` | 0.6 | **0.6** | Igual — se revirtió de 0.5 (workaround innecesario: split ankle H1-2 ayuda tracking, no lo perjudica) |
| `training_curriculum` intervalo | 100 iters | **150 iters** | H1-2 es más complejo (27 DOF). 100 → colapso. 400 → disturb sube cuando penalties ya al máximo. 200 → disturb lento. 150 es el balance. |
| `disturb_rad_scale` | no existe (H1 uniforme implícito) | **[1.0] × 14** uniforme | Revertido de 0.29 para wrists — era workaround para oscilación, ahora manejada por `action_rate_wrist` |
| `PROPRIOCEPTION_DIM` | 63 | 87 | Necesario por 27 DOFs |
| `disturb_dim` | 8 | 14 | Necesario por 14 arm DOFs |
| `arm_start_idx` | hardcoded 11 | dinámico | Necesario |

**Principio de diseño:** los rewards de muñeca (`wrist_deviation`, `action_rate_wrist`) son los únicos añadidos funcionales respecto a h1int. Todo lo demás es adaptación de índices o parámetros para H1-2. `dof_vel_limits` se modifica estructuralmente para reflejar que las muñecas no tienen función locomotora — equivalente a cómo h1int no las tiene en absoluto.

**Mejora futura pendiente — `dof_vel_limits` con `mean` en lugar de `sum`:**
El reward actual usa `torch.sum(error_per_joint)` → el valor total escala con el número de joints. Usar `torch.mean` haría el reward independiente del número de joints (eliminaría `lim_scale`). Probar cuando se añadan manos o si se reconsidera incluir wrists.

---

## 20. Proyecto Caso 4 — ARS vs PPO (H1-2 walk)

Comparación de ARS (Augmented Random Search, Mania et al. 2018) vs PPO para locomoción bipedal H1-2 con brazos fijos.

### 20.1 Tarea h1_2walk

- **DOF controlados:** 13 (piernas ×12 + torso ×1). Brazos (DOF 13-26) fijados a `default_dof_pos` con `_compute_torques` paddeo.
- **Obs:** 50-dim (45 prop + 3 cmd + 2 clock). Sin historial ni info privilegiada.
- **Terreno:** plano (`mesh_type='plane'`). Sin curriculum de terreno ni de penalidades.
- **Archivos:** `h1_2_walk.py`, `h1_2_walk_config.py`, `train_walk.py`
- **`randomize_control_latency=False`**: el buffer de latencia esperaría 60 dims (6+27×2) pero obs es 50 → mismatch. Desactivado explícitamente.

### 20.2 ARS — política lineal

`a = M · normalize(obs)`, M de shape (13, 50) = 650 parámetros.

**Hiperparámetros actuales (v2 post-fix):**

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| `num_pairs` N | 60 | → 120 envs en paralelo |
| `elite_pairs` b | 20 | top-b por `max(r+, r-)` |
| `step_size` α | 0.005 | reducido de 0.02 |
| `noise_std` ν | 0.05 | aumentado de 0.025 |
| `sigma_r_min` | 1.0 | floor crítico (ver §20.3) |
| `max_policy_norm` | 10.0 | hard cap |
| `max_iterations` | 3000 | |

**Update rule (ARS-V2):**
```python
M += (α / (max(sigma_r, sigma_r_min) × ν)) × grad_elite
# donde sigma_r = std(elite rewards), floor evita blowup cuando r+≈r-
```

### 20.3 Bug ARS v1 (resuelto) — sigma_r blowup

**Síntoma:** M_norm creció 7→230 en 939 iters sin mejorar ep_len (48→28 steps).

**Causa:** Con todos los episodios terminando igual de mal (caída en ~30 steps), `sigma_r = std(elite_rews) ≈ 0.3` → escala efectiva = α/(σ·ν) = 0.02/(0.3×0.025) = **2.67×** por iter. La matriz M creció en dirección aleatoria.

**Fix aplicado:** `sigma_r_min=1.0` en `LinearPolicy.update()` (ars.py). Con floor, la escala efectiva cae a 0.005/(1.0×0.05) = **0.10×** — 27× más pequeño. Además `max_policy_norm=10.0` impide que M crezca indefinidamente.

### 20.4 Comparación de eficiencia ARS vs PPO

| | ARS | PPO |
|--|-----|-----|
| Envs | 120 (=2N) | 4096 |
| Update cada | 1000 steps × 120 envs = 120k env-steps | 24 steps × 4096 envs = 98k env-steps |
| Updates/iter | 1 (gradient step) | 5 epochs × (120k/24) ≈ 25k minibatch steps |
| Política | Lineal M (13×50 = 650 params) | MLP [512,256,128] (~200k params) |
| Value fn | No (usa retorno completo del episodio) | Sí (bootstrap con V(s)) |
| Cold start | Vulnerable (sin V(s), necesita episodios completos) | Robusto (`init_at_random_ep_len=True`) |

ARS puede competir con PPO en políticas simples porque la política lineal tiene muy pocos parámetros — cada iteración ARS actualiza los mismos 650 parámetros que PPO actualiza millones de veces.
