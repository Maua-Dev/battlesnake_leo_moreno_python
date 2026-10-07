"""Estrategia para Battlesnake Standard, sem dependencias adicionais."""
from collections import deque
from dataclasses import dataclass
from itertools import product
import logging
from time import perf_counter

from .models import GameState, MoveResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DIRECTIONS = ("up", "down", "left", "right")
WIN = 1_000_000_000
INF = 10**15


def info() -> dict:
    return {
        "apiversion": "1",
        "author": "",
        "color": "#FFBF00",
        "head": "gamer",
        "tail": "mlh-gene",
        "version": "2.0.0",
    }


def start(state: GameState) -> None:
    logger.info("JOGO COMECOU: %s", state.game.id)


def end(state: GameState) -> None:
    logger.info("FIM DE JOGO: %d turnos", state.turn)


@dataclass(frozen=True, slots=True)
class _Snake:
    body: tuple[int, ...]
    health: int


@dataclass(frozen=True, slots=True)
class _Position:
    snakes: tuple[_Snake | None, ...]
    food: frozenset[int]
    ply: int = 0


class _Deadline(Exception):
    pass


class _Engine:
    def __init__(self, state: GameState):
        self.w, self.h = state.board.width, state.board.height
        self.n = self.w * self.h
        self.neighbors = []

        for p in range(self.n):
            x, y = p % self.w, p // self.w

            self.neighbors.append((
                p + self.w if y + 1 < self.h else -1,
                p - self.w if y else -1,
                p - 1 if x else -1,
                p + 1 if x + 1 < self.w else -1,
            ))

        # Hazards sao evitados como obstaculos.
        # O foco desta estrategia e o modo Standard.
        self.hazards = {self.cell(c) for c in state.board.hazards}

        snakes = [state.you] + [
            s for s in state.board.snakes
            if s.id != state.you.id
        ]

        self.initial = _Position(
            tuple(
                _Snake(
                    tuple(self.cell(c) for c in s.body)
                    or (self.cell(s.head),),
                    s.health,
                )
                for s in snakes
            ),
            frozenset(
                self.cell(c)
                for c in state.board.food
                if self.cell(c) >= 0
            ),
        )

        self.had_enemies = len(snakes) > 1

        # Reserva tempo para serializacao, Lambda e rede.
        timeout = state.game.timeout or 500
        budget_ms = max(0, min(160, timeout - 180))
        self.deadline = perf_counter() + budget_ms / 1000

    def cell(self, c) -> int:
        if 0 <= c.x < self.w and 0 <= c.y < self.h:
            return c.y * self.w + c.x

        return -1

    def blocked(self, pos):
        # A ultima parte da cauda sai neste turno.
        # Se a cauda estiver duplicada, a penultima parte ainda bloqueia.
        return self.hazards | {
            p
            for s in pos.snakes
            if s
            for p in s.body[:-1]
            if p >= 0
        }

    def legal(self, pos, i):
        s = pos.snakes[i]

        if s is None or s.body[0] < 0:
            return []

        occupied = self.blocked(pos)
        neck = s.body[1] if len(s.body) > 1 else -1

        return [
            d
            for d, p in enumerate(self.neighbors[s.body[0]])
            if (
                p >= 0
                and p not in occupied
                and p != neck
                and (s.health > 1 or p in pos.food)
            )
        ]

    def emergency(self, pos, i):
        s = pos.snakes[i]

        if s is None or s.body[0] < 0:
            return 0

        occupied = self.blocked(pos)
        neck = s.body[1] if len(s.body) > 1 else -1

        return max(
            range(4),
            key=lambda d: (
                self.neighbors[s.body[0]][d] >= 0,
                self.neighbors[s.body[0]][d] != neck,
                self.neighbors[s.body[0]][d] not in occupied,
            ),
        )

    def advance(self, s, d, food):
        head = (
            self.neighbors[s.body[0]][d]
            if s.body[0] >= 0
            else -1
        )

        body = (head,) + s.body[:-1]

        # No Standard, primeiro move e depois duplica a nova cauda.
        if head in food:
            return _Snake(body + (body[-1],), 100)

        return _Snake(body, s.health - 1)

    def turn(self, pos, moves):
        moved = tuple(
            self.advance(s, moves[i], pos.food) if s else None
            for i, s in enumerate(pos.snakes)
        )

        # Todas comem antes de resolver as colisoes.
        eaten = {
            s.body[0]
            for s in moved
            if s and s.body[0] in pos.food
        }

        active = [
            (
                s is not None
                and s.health > 0
                and s.body[0] >= 0
                and s.body[0] not in self.hazards
            )
            for s in moved
        ]

        alive = active.copy()

        # As eliminacoes por colisao sao simultaneas.
        for i, s in enumerate(moved):
            if not active[i]:
                continue

            for j, other in enumerate(moved):
                if not active[j]:
                    continue

                body_collision = s.body[0] in other.body[1:]

                head_collision = (
                    i != j
                    and s.body[0] == other.body[0]
                    and len(s.body) <= len(other.body)
                )

                if body_collision or head_collision:
                    alive[i] = False
                    break

        return _Position(
            tuple(
                s if alive[i] else None
                for i, s in enumerate(moved)
            ),
            pos.food - eaten,
            pos.ply + 1,
        )

    def distances(self, start, occupied, release=None):
        dist = [self.n + 1] * self.n

        if start < 0:
            return dist

        dist[start] = 0
        queue = deque([start])

        while queue:
            p = queue.popleft()
            t = dist[p] + 1

            for q in self.neighbors[p]:
                if (
                    q < 0
                    or q in self.hazards
                    or dist[q] <= t
                ):
                    continue

                if release is None:
                    if q in occupied:
                        continue
                elif release[q] > t:
                    continue

                dist[q] = t
                queue.append(q)

        return dist

    def evaluate(self, pos):
        me = pos.snakes[0]
        enemies = [s for s in pos.snakes[1:] if s]

        if me is None:
            return (
                -WIN if enemies else -WIN // 2
            ) + pos.ply * 1000

        if self.had_enemies and not enemies:
            return WIN - pos.ply * 1000

        return self.heuristic(pos)

    def heuristic(self, pos):
        me = pos.snakes[0]
        enemies = [s for s in pos.snakes[1:] if s]

        occupied = self.blocked(pos)
        mine = self.distances(me.body[0], occupied)

        theirs = [
            self.distances(s.body[0], occupied)
            for s in enemies
        ]

        length = len(me.body)
        longest = max(
            (len(s.body) for s in enemies),
            default=length,
        )

        lead = length - longest
        space = sum(d <= self.n for d in mine)

        score = (
            min(4, max(-8, lead)) * 2200
            + min(space, length * 2 + 8) * 35
        )

        # A liberacao temporal e uma estimativa.
        # A busca posterior simula os corpos reais.
        if space < length + 2:
            release = [0] * self.n

            for s in pos.snakes:
                if s:
                    for k, p in enumerate(s.body):
                        if p >= 0:
                            release[p] = max(
                                release[p],
                                len(s.body) - k,
                            )

            dynamic = self.distances(
                me.body[0],
                occupied,
                release,
            )

            dynamic_space = sum(
                d <= self.n for d in dynamic
            )

            score -= max(0, length + 2 - space) * 1500

            if dynamic_space < length:
                score -= (
                    2_000_000
                    + (length - dynamic_space) * 10000
                )

        # Territorio: casas onde chegamos antes dos adversarios.
        for p, d in enumerate(mine):
            if d > self.n:
                continue

            if all(
                d < e[p]
                or (
                    d == e[p]
                    and length > len(enemies[k].body)
                )
                for k, e in enumerate(theirs)
            ):
                score += 12

        food_distance = self.n + 1

        for p in pos.food:
            d = mine[p]

            if d > self.n:
                continue

            # Evita depender de comida que o adversario pode pegar antes.
            contested = any(
                e[p] < d
                or (
                    e[p] == d
                    and len(enemies[k].body) >= length
                )
                for k, e in enumerate(theirs)
            )

            food_distance = min(
                food_distance,
                d + (5 if contested else 0),
            )

        hunger = (
            1100 if me.health <= 25
            else 260 if me.health <= 55
            else 90 if lead < 3
            else 12
        )

        score -= min(food_distance, 30) * hunger
        score += me.health * 3

        # Sem comida visivel, nao presume que jamais surgira outra.
        if (
            pos.food
            and me.health <= 25
            and food_distance >= me.health
        ):
            score -= 1_000_000

        for s, distance in zip(enemies, theirs):
            enemy_space = sum(
                d <= self.n for d in distance
            )

            if enemy_space < len(s.body):
                score += 3500

            # Aproxima-se de adversarios quando temos vantagem e espaco.
            if (
                lead > 0
                and space >= length + 2
                and me.health > 35
            ):
                a, b = me.body[0], s.body[0]

                score -= (
                    abs(a % self.w - b % self.w)
                    + abs(a // self.w - b // self.w)
                ) * 25

        p = me.body[0]

        # Preferencia pequena pelo centro, usada como desempate.
        score -= (
            abs(2 * (p % self.w) - self.w + 1)
            + abs(2 * (p // self.w) - self.h + 1)
        )

        return score

    def root_order(self, pos):
        legal = self.legal(pos, 0)
        ranked = []

        for d in legal:
            me = self.advance(pos.snakes[0], d, pos.food)
            head = me.body[0]

            risk = any(
                (
                    s
                    and len(s.body) + (head in pos.food)
                    >= len(me.body)
                    and any(
                        self.neighbors[s.body[0]][m] == head
                        for m in self.legal(pos, i)
                    )
                )
                for i, s in enumerate(pos.snakes)
                if i > 0
            )

            preview = _Position(
                (me,) + pos.snakes[1:],
                pos.food - {head},
                pos.ply + 1,
            )

            ranked.append((
                not risk,
                self.heuristic(preview),
                d,
            ))

        # Evita disputa fatal de cabecas se houver alternativa.
        safe = [r for r in ranked if r[0]]

        return [
            r[2]
            for r in sorted(safe or ranked, reverse=True)
        ]

    def check_time(self):
        if perf_counter() >= self.deadline:
            raise _Deadline

    def maximize(self, pos, depth, alpha, beta):
        self.check_time()

        if (
            depth == 0
            or pos.snakes[0] is None
            or (
                self.had_enemies
                and not any(pos.snakes[1:])
            )
        ):
            return self.evaluate(pos)

        best = -INF

        for d in (
            self.legal(pos, 0)
            or [self.emergency(pos, 0)]
        ):
            best = max(
                best,
                self.minimize(pos, d, depth, alpha, beta),
            )

            alpha = max(alpha, best)

            if alpha >= beta:
                break

        return best

    def minimize(self, pos, move, depth, alpha, beta):
        choices = [
            (
                self.legal(pos, i)
                or [self.emergency(pos, i)]
            )
            if s else [0]
            for i, s in enumerate(pos.snakes)
            if i > 0
        ]

        worst = INF

        # Considera todas as combinacoes de movimentos adversarios.
        for combo in product(*choices):
            self.check_time()

            child = self.turn(pos, (move,) + combo)

            worst = min(
                worst,
                self.maximize(
                    child,
                    depth - 1,
                    alpha,
                    beta,
                ),
            )

            beta = min(beta, worst)

            if alpha >= beta:
                break

        return worst

    def choose(self):
        pos = self.initial
        order = self.root_order(pos)

        if not order:
            return self.emergency(pos, 0)

        chosen = order[0]

        if len(order) == 1:
            return chosen

        # Somente uma profundidade completa pode atualizar a escolha.
        for depth in range(1, 9):
            best, value, alpha = chosen, -INF, -INF

            try:
                for d in order:
                    score = self.minimize(
                        pos,
                        d,
                        depth,
                        alpha,
                        INF,
                    )

                    if score > value:
                        best, value = d, score

                    alpha = max(alpha, value)

            except _Deadline:
                break

            # Se todas parecem perder, conserva a escolha anterior.
            if value <= -WIN // 4:
                break

            chosen = best

            order = [chosen] + [
                d for d in order if d != chosen
            ]

            if value >= WIN // 2:
                break

        return chosen


def get_move(state: GameState) -> MoveResponse:
    if (
        state.board.width <= 0
        or state.board.height <= 0
        or state.board.width * state.board.height > 2500
    ):
        return MoveResponse(move="up")

    engine = _Engine(state)
    direction = DIRECTIONS[engine.choose()]

    logger.info("MOVE %d: %s", state.turn, direction)

    return MoveResponse(move=direction)