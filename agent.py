import random
import math
import numpy as np
import pygame
from dynamicBrain import DynamicBrain
from settings import (
    WIDTH, HEIGHT, MUTATION_RATE,
    RAY_COUNT, RAY_SPREAD, RAY_LENGTH, EXTRA_INPUTS, PRED_REPRO_THRESHOLD,PREY_REPRO_THRESHOLD
)
import torch
from rl_head import RLHead
from settings import (
    DESIRED_TRAITS, GAMMA, REWARD_REPRO_THRESHOLD
)


class Agent:
    predator_imgs = []
    prey_imgs     = []


    def __init__(self, x, y, brain=None, traits=None, role=None, mutation_rate=None, is_human_controlled = False):
        INPUT_SIZE = RAY_COUNT * 3
        HIDDEN_SIZE = 16  # İstersen ayarlanabilir yaparız
        self.x, self.y    = x, y
        self.angle        = random.uniform(-math.pi/6, math.pi/6)
        self.energy       = 200.0
        self.role         = role or random.choice(["predator", "prey"])
        self.mutation_rate = mutation_rate if mutation_rate is not None else MUTATION_RATE
        self.age = 0
        self.children_count = 0
        self.is_human_controlled = is_human_controlled
        self.frame_index = 0
        self.anim_timer  = 0
        self.rl_head = RLHead()
        self.saved_log_probs = []
        self.rewards         = []
        self.total_reward    = 0.0
        

        # Genetic traits: speed, rotation_speed, efficiency
        if traits is None:
            self.speed = random.uniform(1.5, 3.0) if self.role == "prey" else random.uniform(2.0, 4.0) 
            self.rotation_speed = 0.2
            self.efficiency = 0.5 if self.role == "prey" else 1.5
            self.vision_range = random.uniform(60, 120)  # yeni gen
            self.risk_tolerance = random.uniform(0.0, 1.0)  # yeni gen
            self.grouping_preference = random.uniform(0.0, 1.0)  # yeni gen
        else:
            speed, rot, eff, vis, risk, group = traits
            self.speed = np.clip(speed + (random.random() < MUTATION_RATE) * np.random.normal(0, 0.3), 0.5, 5.0) if self.role == "prey" else np.clip(speed + (random.random() < MUTATION_RATE) * np.random.normal(0, 0.3), 1, 7.0)
            self.rotation_speed = rot
            self.efficiency = np.clip(eff + (random.random() < MUTATION_RATE) * np.random.normal(0, 0.1), 0.5, 2.0)
            self.vision_range = np.clip(vis + (random.random() < MUTATION_RATE) * np.random.normal(0, 5.0), 40, 160)
            self.risk_tolerance = np.clip(risk + (random.random() < MUTATION_RATE) * np.random.normal(0, 0.1), 0.0, 1.0)
            self.grouping_preference = np.clip(group + (random.random() < MUTATION_RATE) * np.random.normal(0, 0.1), 0.0, 1.0)



        

        self.brain = brain if brain is not None else DynamicBrain(
            input_size=RAY_COUNT * 3 + EXTRA_INPUTS ,
            output_size=2,
            mutation_rate=self.mutation_rate if mutation_rate is not None
                                                  else MUTATION_RATE
        )



    def human_controlled_movement(self, keys):
        if keys[pygame.K_w]:  # İleri
            self.x += math.cos(self.angle) * self.speed
            self.y += math.sin(self.angle) * self.speed
            self.energy -= 0.05 * (1/self.efficiency)
        if keys[pygame.K_a]:  # Sola dön
            self.angle -= self.rotation_speed
            self.energy -= 0.02 * (1/self.efficiency)
        if keys[pygame.K_d]:  # Sağa dön
            self.angle += self.rotation_speed
            self.energy -= 0.02 * (1/self.efficiency)

        # Sınırları koru
        SIM_WIDTH = WIDTH-250
        self.x %= SIM_WIDTH
        self.y %= HEIGHT

    def sense(self, food_list, agent_list, grid):
      

        cx = int(self.x) // grid.cell_size
        cy = int(self.y) // grid.cell_size
        nearby_food = []
        nearby_agents = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (cx + dx, cy + dy)
                if cell in grid.cells:
                    nearby_food.extend(grid.cells[cell]['food'])
                    nearby_agents.extend(grid.cells[cell]['agents'])

        if self.role == 'predator':
            angle_offsets = np.linspace(-RAY_SPREAD/2, RAY_SPREAD/2, RAY_COUNT)
        else:
            angle_offsets = np.linspace(0, 2 * math.pi, RAY_COUNT, endpoint=False)

        inputs = np.zeros(RAY_COUNT * 3 + EXTRA_INPUTS, dtype=float)
        self.rays = []

        for i, offset in enumerate(angle_offsets):
            ray_angle = self.angle + offset
            sx, sy = self.x, self.y
            ex = sx + math.cos(ray_angle) * self.vision_range
            ey = sy + math.sin(ray_angle) * self.vision_range
            self.rays.append(((sx, sy), (ex, ey)))

            def dist_pt_seg(px, py, ax, ay, bx, by):
                dx, dy = bx - ax, by - ay
                if dx == 0 and dy == 0:
                    return math.hypot(px - ax, py - ay)
                t = ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
                t = max(0.0, min(1.0, t))
                cx_, cy_ = ax + t * dx, ay + t * dy
                return math.hypot(px - cx_, py - cy_)

            for f in nearby_food:
                dist = dist_pt_seg(f.x, f.y, sx, sy, ex, ey)
                if dist < f.size:
                    proximity = 1.0 - (dist / self.vision_range)
                    inputs[i * 3 + 0] = max(inputs[i * 3 + 0], proximity)

            for other in nearby_agents:
                if other is self:
                    continue
                dist = dist_pt_seg(other.x, other.y, sx, sy, ex, ey)
                if dist < 8:
                    proximity = 1.0 - (dist / self.vision_range)
                    if other.role == 'prey':
                        inputs[i * 3 + 1] = max(inputs[i * 3 + 1], proximity)
                    else:
                        inputs[i * 3 + 2] = max(inputs[i * 3 + 2], proximity)

            

        # Alignment / Cohesion / Separation
        align_x, align_y = 0.0, 0.0
        cohesion_x, cohesion_y = 0.0, 0.0
        min_separation = float('inf')
        count = 0

        for other in nearby_agents:
            if other is not self and other.role == self.role:
                dx = other.x - self.x
                dy = other.y - self.y
                dist = math.hypot(dx, dy)
                if dist < self.vision_range:
                    align_x += math.cos(other.angle)
                    align_y += math.sin(other.angle)
                    cohesion_x += other.x
                    cohesion_y += other.y
                    if dist < min_separation:
                        min_separation = dist
                    count += 1

        g = self.grouping_preference
        scale = 1.0
        if count > 0:
            align_norm = math.hypot(align_x, align_y)
            inputs[-5] = scale* g * align_x / align_norm if align_norm != 0 else 0.0
            inputs[-4] = scale* g * align_y / align_norm if align_norm != 0 else 0.0
            avg_x = cohesion_x / count
            avg_y = cohesion_y / count
            dx = (avg_x - self.x) / self.vision_range
            dy = (avg_y - self.y) / self.vision_range
            inputs[-3] = scale* np.clip(g * dx, -1, 1)
            inputs[-2] = scale* np.clip(g * dy, -1, 1)
            inputs[-1] = scale* min(1.0, g * min_separation / self.vision_range)
        else:
            inputs[-5:] = 0.0, 0.0, 0.0, 0.0, 1.0

        # Enerji inputu
        max_energy = PREY_REPRO_THRESHOLD if self.role == "prey" else PRED_REPRO_THRESHOLD
        inputs[RAY_COUNT * 3] = min(1.0, self.energy / max_energy)

        return inputs


    def _calc_reward(self, energy_delta):
        d = DESIRED_TRAITS[self.role]
        r  = 0.0
        # trait ödülleri
        r += 2.0 * (1.0 - abs(self.speed - d["speed_target"])   / 5.0)
        r += 1.5 * (1.0 - abs(self.vision_range - d["vision_target"]) / 160)
        r += self.efficiency       # 0.5 – 2.0
        # yaşam-bazlı
        r += 0.02 * energy_delta
        r += 0.5                   # “yaşıyorum” primi
        return r

    def think(self, inputs):
        # DynamicBrain hem eylem, hem de RL-gizli vektör döndürsün
        dv_raw, dth_raw, hvec = self.brain.think(inputs, return_hidden=True)
        dist = self.rl_head(hvec)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action).sum())
        # çıktı -1…1 aralığında kalsın
        dv     = float(torch.clamp(action[0], -1, 1))
        dtheta = float(torch.clamp(action[1], -1, 1))
        return dv, dtheta




        # ================================================================= #
    #  Agent.update —— POPÜLASYON DÖNGÜSÜNDE HER KARE ÇAĞRILIR          #
    # ================================================================= #
    def update(self, food_list, agents, grid, keys=None):
        """
        • Sensör → beyin + RL-policy → hareket  
        • Enerji maliyetleri / beslenme / mutasyon  
        • Ödül hesabı (RL)  
        • Ölüm durumunda REINFORCE güncellemesi
        """
        from settings import (WIDTH, HEIGHT,
                              SPEED_PENALTY_THRESHOLD, SPEED_PENALTY_RATE,
                              VISION_PENALTY_THRESHOLD, VISION_PENALTY_RATE)

        self.age += 1
        prev_energy = self.energy            # ödül için

        # -------------------------------------------------------------- #
        # 1) HAREKET (insan kontrolü varsa klavye, yoksa beyin + RL)     #
        # -------------------------------------------------------------- #
        if self.is_human_controlled and keys is not None:
            self.human_controlled_movement(keys)

        else:
            inputs = self.sense(food_list, agents, grid)
            dv, dtheta = self.think(inputs)          # RL-policy

            # hız faktörü
            move_factor = dv if dv >= 0 else 0.4 * dv
            # yön güncelle
            self.angle += dtheta * self.rotation_speed
            # pozisyon güncelle
            self.x += math.cos(self.angle) * self.speed * move_factor
            self.y += math.sin(self.angle) * self.speed * move_factor

            # hareket enerjisi
            self.energy -= 0.5 * (abs(dv) * 0.04 + abs(dtheta) * 0.01) / self.efficiency
            # avcı sabit cezası
            if self.role == "predator":
                self.energy -= 0.1

        # dünya kenarını döngüsel yap
        SIM_WIDTH = WIDTH - 250
        self.x %= SIM_WIDTH
        self.y %= HEIGHT

        # -------------------------------------------------------------- #
        # 2) BESLENME / AVLANMA                                          #
        # -------------------------------------------------------------- #
        if self.role == "prey":
            for f in food_list:
                if math.hypot(self.x - f.x, self.y - f.y) < f.size:
                    self.energy += 50
                    self.mutation_rate = max(0.03, self.mutation_rate * 0.98)
                    self.brain.mutation_rate = self.mutation_rate
                    food_list.remove(f)
                    break
        else:  # predator
            for other in agents:
                if other is not self and other.role == "prey":
                    if math.hypot(self.x - other.x, self.y - other.y) < 8:
                        self.energy += 150
                        self.mutation_rate = max(0.3, self.mutation_rate * 0.96)
                        self.brain.mutation_rate = self.mutation_rate
                        agents.remove(other)
                        break

        # -------------------------------------------------------------- #
        # 3) ANİMASYON TİKİ & STRES-BAĞLI MUTASYON                        #
        # -------------------------------------------------------------- #
        self.anim_timer += 1
        if self.anim_timer >= 10:
            self.anim_timer = 0
            self.frame_index ^= 1

        if self.energy < 70:                       # stres → mutasyon ↑
            self.mutation_rate = min(1.0, self.mutation_rate * 1.01)
            self.brain.mutation_rate = self.mutation_rate

        # küçük on-line mutasyon olasılıkları
        if random.random() < self.mutation_rate:
            self.speed = np.clip(self.speed + np.random.normal(0, 0.02), 0.5, 5.0)
        if random.random() < self.mutation_rate:
            self.efficiency = np.clip(self.efficiency + np.random.normal(0, 0.01), 0.5, 2.0)

        # enerjiden bağımsız “online” ağırlık oynatma
        # (daha düşük olasılık – başarıya göre ayarlandı)
        if random.random() < self.mutation_rate * 0.3:
            self.brain.mutate_small()

        # hız & görüş maliyetleri
        if self.speed > SPEED_PENALTY_THRESHOLD:
            excess = self.speed - SPEED_PENALTY_THRESHOLD
            self.energy -= SPEED_PENALTY_RATE * excess

        if self.vision_range > VISION_PENALTY_THRESHOLD:
            excess = self.vision_range - VISION_PENALTY_THRESHOLD
            self.energy -= VISION_PENALTY_RATE * excess

        # -------------------------------------------------------------- #
        # 4) RL ÖDÜLÜ HESAPLA                                            #
        # -------------------------------------------------------------- #
        energy_delta = self.energy - prev_energy
        r = self._calc_reward(energy_delta)        # trait + enerji + yaşama
        self.rewards.append(r)
        self.total_reward += r

        # -------------------------------------------------------------- #
        # 5) ÖLÜM KONTROLÜ → EPISODE BİTİŞİ + GRADIENT UPDATE            #
        # -------------------------------------------------------------- #
        if self.energy <= 0:
            self._finish_episode()                 # REINFORCE
            # Ana döngü bu ajanı listeden zaten çıkaracak
    # ================================================================= #

    def _finish_episode(self):
        if not self.rewards:      # koruma
            return
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + GAMMA*R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std()+1e-9)
        loss = 0
        for logp, G in zip(self.saved_log_probs, returns):
            loss += -logp * G
        self.rl_head.opt.zero_grad()
        loss.backward()
        self.rl_head.opt.step()
        # buffer’ı temizle
        self.rewards.clear()
        self.saved_log_probs.clear()
        self.total_reward = 0.0
    def ready_to_reproduce(self, thr_energy, thr_reward):
        return (self.energy > thr_energy) and (self.total_reward > thr_reward)
    def draw(self, surface, selected = False):
        imgs = Agent.predator_imgs if self.role=='predator' else Agent.prey_imgs
        img  = imgs[self.frame_index]
        ang_deg = -math.degrees(self.angle)
        rt   = pygame.transform.rotate(img, ang_deg)
        rect = rt.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rt, rect)
        # Eğer seçiliyse etrafına sarı daire çizelim
        if selected:
            pygame.draw.circle(surface, (255, 255, 0), (int(self.x), int(self.y)), 12, 2)
            for start, end in getattr(self, "rays", []):
                pygame.draw.line(surface, (0, 255, 255), start, end, 1)
    def fitness(self):
        return self.energy + self.age * 0.5 + self.children_count * 50
    def to_dict(self):
        return {
            "role": self.role,
            "speed": self.speed,
            "rotation_speed": self.rotation_speed,
            "efficiency": self.efficiency,
            "vision_range": self.vision_range,
            "mutation_rate": self.mutation_rate,
            "risk_tolerance": self.risk_tolerance,
            "grouping_preference": self.grouping_preference,
            "brain": self.brain.to_dict()
        }


    @classmethod
    def from_dict(cls, data):
        brain = DynamicBrain.from_dict(data["brain"])
        agent = cls(
            x=random.random() * (WIDTH - 250),
            y=random.random() * HEIGHT,
            traits=(
                data["speed"],
                data["rotation_speed"],
                data["efficiency"],
                data["vision_range"],
                data["risk_tolerance"],
                data["grouping_preference"]
            ),
            role=data["role"],
            mutation_rate=data["mutation_rate"]
        )
        agent.brain = brain
        return agent


