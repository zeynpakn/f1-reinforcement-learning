import pygame
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS,
    MAX_STEPS_PER_EPISODE, HEADLESS,
    REWARD_ALIVE, REWARD_CHECKPOINT,
    REWARD_CRASH, REWARD_SLOW,
    REWARD_FORWARD, REWARD_BACKWARD
)
from car import Car
from track import Track
from sensor import SensorSystem

class Environment:
    def __init__(self, headless=HEADLESS):
        self.headless = headless

        pygame.init()
        if self.headless:
            import os
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.screen = pygame.display.set_mode((1, 1))
        else:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("F1 RL - Training")

        self.clock   = pygame.time.Clock()
        self.track   = Track()
        self.car     = Car(self.track.start_x,
                           self.track.start_y,
                           self.track.start_angle)
        self.sensors = SensorSystem()

        self.next_checkpoint = 0
        self.steps           = 0
        self.total_reward    = 0
        self.lap_count       = 0

    # ── Sıfırla ────────────────────────────────────────
    def reset(self):
        self.car.reset(self.track.start_x,
                       self.track.start_y,
                       self.track.start_angle)
        self.sensors.update(self.car.x, self.car.y,
                            self.car.angle, self.track)
        self.next_checkpoint = 0
        self.steps           = 0
        self.total_reward    = 0
        return self._get_state()

    # ── Bir adım ───────────────────────────────────────
    def step(self, action):
        self._apply_action(action)
        self.car.update()
        self.sensors.update(self.car.x, self.car.y,
                            self.car.angle, self.track)
        self.steps += 1

        reward, done = self._calculate_reward()
        self.total_reward += reward
        state = self._get_state()

        info = {
            "checkpoint": self.next_checkpoint,
            "lap":        self.lap_count,
            "steps":      self.steps,
            "speed":      self.car.speed,
        }
        return state, reward, done, info

    # ── Aksiyon uygula ─────────────────────────────────
    def _apply_action(self, action):
        # 0→düz, 1→sol, 2→sağ, 3→hızlan, 4→yavaşla
        if action == 0:
            self.car.accelerate()
        elif action == 1:
            self.car.turn_left()
            self.car.accelerate()
        elif action == 2:
            self.car.turn_right()
            self.car.accelerate()
        elif action == 3:
            self.car.accelerate()
            self.car.accelerate()
        elif action == 4:
            self.car.brake()

    # ── Reward hesapla ─────────────────────────────────
    def _calculate_reward(self):
        done = False

        # Çarpışma
        corners = self.car.get_corners()
        if self.track.check_corners(corners):
            return REWARD_CRASH, True

        # Checkpoint
        if self.track.check_checkpoint(self.car.x, self.car.y,
                                        self.next_checkpoint):
            self.next_checkpoint += 1
            if self.next_checkpoint >= self.track.checkpoint_count:
                self.next_checkpoint = 0
                self.lap_count += 1
            return REWARD_CHECKPOINT, False

        # Hız bazlı reward
        if self.car.speed <= 0:
            reward = REWARD_BACKWARD
        elif self.car.speed < 1.0:
            reward = REWARD_SLOW
        else:
            reward = REWARD_ALIVE + REWARD_FORWARD

        # Max adım
        if self.steps >= MAX_STEPS_PER_EPISODE:
            done = True

        return reward, done

    # ── State ──────────────────────────────────────────
    def _get_state(self):
        normalized = self.sensors.get_normalized()
        speed_norm = self.car.speed / 6.0
        return tuple(normalized) + (speed_norm,)

    # ── Render ─────────────────────────────────────────
    def render(self):
        if self.headless:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.track.draw(self.screen)
        self.car.draw(self.screen)
        self.sensors.draw(self.screen, self.car.x, self.car.y)
        pygame.display.flip()
        self.clock.tick(FPS)