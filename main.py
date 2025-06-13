import pygame
import random
from settings import (
    WIDTH, HEIGHT, FPS, NUM_AGENTS, NUM_FOOD, FOOD_RESPAWN_RATE,
    GRID_SIZE, WHITE, GREEN, BEIGE, BLACK, BLUE,RED,REWARD_REPRO_THRESHOLD
)
from agent import Agent
from food import Food
from button import Button
import math



def draw_agent_inspector(surface, agent, font):
    """Sağda bir bölme açıp ajan bilgilerini yazar."""
    panel_x = 1450
    panel_w = WIDTH - panel_x
    panel_h = HEIGHT
    
    pygame.draw.rect(surface, BEIGE, (panel_x, 0, panel_w, panel_h))
    pygame.draw.rect(surface, BLACK, (panel_x, 0, panel_w, panel_h), 2)

    lines = []
    lines.append(f"Agent Details:")
    lines.append(f" Role:       {agent.role}")
    lines.append(f" Speed:      {agent.speed:.2f}")
    lines.append(f" Efficiency: {agent.efficiency:.2f}")
    lines.append(f" Current Energy: {agent.energy:.2f}")
    lines.append(f" Mutation Rate: {agent.mutation_rate:.2f}")
    lines.append(f" Vision:     {agent.vision_range:.1f}")
    lines.append(f" RiskTol:    {agent.risk_tolerance:.2f}")
    lines.append(f" GroupPref:  {agent.grouping_preference:.2f}")

    lines.append("")
    


    for idx, text in enumerate(lines):
        surf = font.render(text, True, BLACK)
        surface.blit(surf, (panel_x + 10, 120 + idx * 20))
    agent.brain.draw(surface, panel_x, panel_w, font)

import json
import os



class SpatialGrid:
    """Her hücreye food ve agent listesi tutan basit spatial-hash."""
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.cells = {}

    def rebuild(self, foods, agents):
        self.cells.clear()
        for f in foods:
            cell = (int(f.x)//self.cell_size, int(f.y)//self.cell_size)
            self.cells.setdefault(cell, {'food':[], 'agents':[]})['food'].append(f)
        for a in agents:
            cell = (int(a.x)//self.cell_size, int(a.y)//self.cell_size)
            self.cells.setdefault(cell, {'food':[], 'agents':[]})['agents'].append(a)

def main():
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Raycast Evolution")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont(None, 24)
    paused = False
    selected_agent = None

    # 1) Load ALL your sprites onto the classes
    Agent.predator_imgs = [
        pygame.image.load("assets/predator1.png").convert_alpha(),
        pygame.image.load("assets/predator2.png").convert_alpha()
    ]
    Agent.prey_imgs = [
        pygame.image.load("assets/prey1.png").convert_alpha(),
        pygame.image.load("assets/prey2.png").convert_alpha()
    ]
    # scale if you want them bigger on screen
    Agent.predator_imgs = [pygame.transform.scale(img, (16,16)) for img in Agent.predator_imgs]
    Agent.prey_imgs     = [pygame.transform.scale(img, (16,16)) for img in Agent.prey_imgs]

    # load your new food sprite (plum.png) *before* creating any Food()
    Food.img = pygame.image.load("assets/plum.png").convert_alpha()
    # optional: Food.img = pygame.transform.scale(Food.img, (16,16))

    # 2) Now create your initial agents + food_list
    agents    = [Agent(random.random()*WIDTH, random.random()*HEIGHT)
                 for _ in range(NUM_AGENTS)]
    food_list = [Food() for _ in range(NUM_FOOD)]
    total_born = len(agents)
    grid = SpatialGrid(GRID_SIZE)
    def save_selected_agent():
        if selected_agent:
            name = input("Agent name to save: ")
            filename = f"saved_agents/{name}.json"
            os.makedirs("saved_agents", exist_ok=True)
            with open(filename, "w") as f:
                json.dump(selected_agent.to_dict(), f, indent=2)
            print(f"Saved agent as {filename}")

    def load_saved_agent():
        filename = input("Agent filename to load (without .json): ")
        full_path = f"saved_agents/{filename}.json"
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                data = json.load(f)
                new_agent = Agent.from_dict(data)
                agents.append(new_agent)
                print(f"Loaded agent {filename} into simulation.")
        else:
            print(f"No such agent file: {full_path}")
    # Butonlar
    padding = 10
    bw, bh = 140, 32
    def add_prey():
        nonlocal total_born
        agents.append(Agent(random.random()*WIDTH, random.random()*HEIGHT, role='prey'))
        total_born += 1

    def add_pred():
        nonlocal total_born
        agents.append(Agent(random.random()*WIDTH, random.random()*HEIGHT, role='predator'))
        total_born += 1
    def save_world(auto=False):
        os.makedirs("saved_worlds", exist_ok=True)
        if auto:
            filename = f"saved_worlds/world_save.json"
        else:
            name = input("Agent name to save: ")
            filename = f"saved_worlds/{name}.json"

        data = {
            "agents": [a.to_dict() for a in agents],
            "food": [{"x": f.x, "y": f.y} for f in food_list],
            "total_born": total_born
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print("World saved to", filename)
        
    def load_world(auto = False):
        nonlocal agents, food_list, total_born
        if auto:
            filename = "saved_worlds/world_save.json"
        else:
            inp = input("world filename to load (without .json): ")
            filename = f"saved_worlds/{inp}.json"
        if not os.path.exists(filename):
            print("No saved world found.")
            return

        with open(filename, "r") as f:
            data = json.load(f)

        agents = [Agent.from_dict(ad) for ad in data["agents"]]
        food_list = []
        for fd in data["food"]:
            f = Food()
            f.x = fd["x"]
            f.y = fd["y"]
            food_list.append(f)

        total_born = data.get("total_born", len(agents))
        print("World loaded from", filename)

    prey_btn = Button(
        rect=(padding, HEIGHT-bh-padding, bw, bh),
        text="Add Prey", font=font, callback=add_prey
    )
    pred_btn = Button(
        rect=(padding*2+bw, HEIGHT-bh-padding, bw, bh),
        text="Add Predator", font=font, callback=add_pred
    )
    save_btn = Button(
        rect=(padding*3+bw*2, HEIGHT-bh-padding, bw, bh),
        text="Save Agent", font=font, callback=save_selected_agent
    )

    load_btn = Button(
        rect=(padding*4+bw*3, HEIGHT-bh-padding, bw, bh),
        text="Load Agent", font=font, callback=load_saved_agent
    )

    buttons = [prey_btn, pred_btn, save_btn, load_btn]
    save_world_btn = Button(
    rect=(padding*5 + bw*4, HEIGHT-bh-padding, bw, bh),
    text="Save World", font=font, callback=save_world
    )

    load_world_btn = Button(
        rect=(padding*6 + bw*5, HEIGHT-bh-padding, bw, bh),
        text="Load World", font=font, callback=load_world
    )

    buttons.extend([save_world_btn, load_world_btn])

    
    from settings import (
        PREY_REPRO_THRESHOLD, PREY_REPRO_COST,
        PRED_REPRO_THRESHOLD, PRED_REPRO_COST
    )
    prey_history = []
    pred_history = []
    avg_age_hist     = []
    avg_energy_hist  = []
    avg_sep_hist     = []

    graph_width  = 200     # px
    graph_height = 100     # px
    graph_x = WIDTH - graph_width - 10
    graph_y = 10
    running = True
    while running:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_e:
                    human_agent = Agent(
                        random.random()*(WIDTH-250),
                        random.random()*HEIGHT,
                        role='prey',
                        is_human_controlled=True
                    )
                    agents.append(human_agent)

            for btn in buttons:
                btn.handle_event(ev)
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused

            if paused and ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:  # Sol tıklama
                    mx, my = ev.pos
                    for a in agents:
                        if math.hypot(a.x - mx, a.y - my) < 10:  # 10 pixel içinde mi?
                            selected_agent = a
                            break


        # Spatial grid’i güncelle
        grid.rebuild(food_list, agents)

        if not paused:
            for a in agents[:]:
                keys = pygame.key.get_pressed()
                a.update(food_list, agents, grid, keys)

                # Üreme eşiğini ve maliyetini role’a göre seç
                if a.role == 'prey':
                    thr, cost = PREY_REPRO_THRESHOLD, PREY_REPRO_COST
                else:
                    thr, cost = PRED_REPRO_THRESHOLD, PRED_REPRO_COST

                # Eğer enerji eşikten fazlaysa üreyip enerji harca
                if a.ready_to_reproduce(thr, REWARD_REPRO_THRESHOLD):
                    a.energy -= cost
                    child = Agent(
                        a.x, a.y,
                        brain=a.brain.clone(),
                        traits=(a.speed, a.rotation_speed, a.efficiency,
                                a.vision_range, a.risk_tolerance,
                                a.grouping_preference),
                        role=a.role,
                        mutation_rate=max(0.03, a.mutation_rate*0.95)   # başarılıysa az mutasyon
                    )
                    agents.append(child)
                    total_born += 1
                    a.children_count += 1
 
                # Ölüm kontrolü
                if a.energy <= 0:
                    agents.remove(a)
                # Yemek yenileme
            if random.random() < FOOD_RESPAWN_RATE:
                food_list.append(Food())
        

        # Çizim
        screen.fill(GREEN)


        for f in food_list:
            f.draw(screen)
        for a in agents:
            a.draw(screen, selected=(a is selected_agent))

        for btn in buttons:
            btn.draw(screen)
        if selected_agent:
            draw_agent_inspector(screen, selected_agent, font)
        

        # İstatistikler
        prey_count = sum(1 for a in agents if a.role=='prey')
        pred_count = sum(1 for a in agents if a.role=='predator')
        avg_energy = sum(a.energy for a in agents)/len(agents) if agents else 0
        # prey_count, pred_count hesaplandığı bölümden hemen sonra
        if agents:
            avg_age_hist.append(sum(a.age for a in agents)/len(agents))
            avg_energy_hist.append(sum(a.energy for a in agents)/len(agents))
            # sürü içi ortalama mesafe (yalnız prey’ler arası)
            prey_pos = [(a.x, a.y) for a in agents if a.role=='prey']
            if len(prey_pos) > 1:
                dsum = 0; n = 0
                for i in range(len(prey_pos)):
                    for j in range(i+1, len(prey_pos)):
                        dx = prey_pos[i][0]-prey_pos[j][0]
                        dy = prey_pos[i][1]-prey_pos[j][1]
                        dsum += math.hypot(dx,dy); n += 1
                avg_sep_hist.append(dsum/n)
            else:
                avg_sep_hist.append(0)
        if pygame.time.get_ticks() % 6000 < 16:               # 60 FPS → 600 kare
            if avg_age_hist:
                print(f"[{len(avg_age_hist)}] age={avg_age_hist[-1]:.1f} "
                    f"energy={avg_energy_hist[-1]:.1f} "
                    f"sep={avg_sep_hist[-1]:.1f}")

        prey_history.append(prey_count)
        pred_history.append(pred_count)
        if len(prey_history) > graph_width:
            prey_history.pop(0)
            pred_history.pop(0)

        # istatistik metinleri
        stats = [
            f"Agents:     {len(agents)}",
            f"Prey:       {prey_count}",
            f"Predator:   {pred_count}",
            f"Avg Energy: {avg_energy:.1f}",
            f"Total Born: {total_born}"
        ]
        for i, line in enumerate(stats):
            surf = font.render(line, True, WHITE)
            screen.blit(surf, (10, 10 + i*20))
        if prey_count == 0 or pred_count == 0:
            load_world(True)
        if total_born % 200 == 0:
            save_world(True)

        # —— buradan itibaren grafik çizimi —— 
        # grafik arka planı
        pygame.draw.rect(screen, BEIGE, (graph_x, graph_y, graph_width, graph_height))
        # çerçeve
        pygame.draw.rect(screen, BLACK, (graph_x, graph_y, graph_width, graph_height), 1)

        # ölçek faktörü (dinamik olarak en büyük değere göre)
        max_val = max(prey_history + pred_history + [1])
        # çizgi listeleri
        prey_points = []
        pred_points = []
        for idx, (py, pd) in enumerate(zip(prey_history, pred_history)):
            x = graph_x + idx
            # y ekseni ters yönde: graph_y + graph_height - değer*ölçek
            y_prey = graph_y + graph_height - int((py / max_val) * (graph_height-1))
            y_pred = graph_y + graph_height - int((pd / max_val) * (graph_height-1))
            prey_points.append((x, y_prey))
            pred_points.append((x, y_pred))

        # çizgileri çiz
        if len(prey_points) > 1:
            pygame.draw.lines(screen, BLUE, False, prey_points, 1)
            pygame.draw.lines(screen, RED,  False, pred_points, 1)
        # —— grafik çizimi bitti —— 

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
