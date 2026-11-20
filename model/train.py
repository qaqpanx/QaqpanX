"""QaqpanX: синтетические данные + модель «дроп / не дроп».

Запуск:  python train.py
Выход:   dataset.csv, model.json (веса для демо), метрики в консоль.

Данные синтетические: реальных банковских данных у нас нет. Специально добавлены
«трудные» честные клиенты (день рождения, продажи на OLX, новые карты, переводы
между своими счетами) и «осторожные» дропы — чтобы модель не была идеальной.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

rng = np.random.default_rng(42)
N = 10_000
DROP_SHARE = 0.03

FEATURES = [
    ("card_age_days",   "Возраст карты, дней"),
    ("senders_24h",     "Отправителей за сутки"),
    ("unknown_share",   "Доля незнакомых отправителей"),
    ("minutes_to_out",  "Минут до вывода дальше"),
    ("outflow_ratio",   "Доля денег, ушедших дальше"),
    ("new_device_city", "Новый телефон в другом городе"),
    ("age",             "Возраст клиента"),
    ("amount_vs_usual", "Сумма к обычной, раз"),
]


def legit(kind):
    if kind == "regular":
        return [rng.uniform(30, 3000), rng.integers(0, 4), rng.uniform(0, .4), rng.uniform(120, 1440),
                rng.uniform(0, .5), rng.random() < .02, rng.integers(18, 75), rng.lognormal(0, .4)]
    if kind == "birthday":   # студенту скинулись друзья
        return [rng.uniform(60, 1500), rng.integers(5, 21), rng.uniform(.2, .7), rng.uniform(60, 1440),
                rng.uniform(0, .6), rng.random() < .03, rng.integers(16, 25), rng.lognormal(1.2, .5)]
    if kind == "seller":     # продаёт вещи на OLX
        return [rng.uniform(30, 2500), rng.integers(3, 11), rng.uniform(.8, 1), rng.uniform(30, 1440),
                rng.uniform(0, .7), rng.random() < .03, rng.integers(18, 60), rng.lognormal(.8, .5)]
    if kind == "newbie":     # только открыл карту
        return [rng.uniform(0, 30), rng.integers(0, 4), rng.uniform(0, .6), rng.uniform(60, 1440),
                rng.uniform(0, .6), rng.random() < .15, rng.integers(16, 70), rng.lognormal(.3, .5)]
    # own: гоняет деньги между своими счетами быстро
    return [rng.uniform(60, 3000), rng.integers(1, 3), rng.uniform(0, .2), rng.uniform(1, 30),
            rng.uniform(.7, 1), rng.random() < .05, rng.integers(20, 60), rng.lognormal(.5, .5)]


def drop(kind):
    if kind == "naive":      # подросток с OLX
        return [rng.uniform(0, 60), rng.integers(2, 16), rng.uniform(.7, 1), rng.uniform(1, 30),
                rng.uniform(.85, .97), rng.random() < .6, rng.integers(16, 23), rng.lognormal(2, .6)]
    # careful: опытный дроп, маскируется
    return [rng.uniform(30, 900), rng.integers(1, 5), rng.uniform(.4, 1), rng.uniform(10, 240),
            rng.uniform(.6, .95), rng.random() < .3, rng.integers(20, 55), rng.lognormal(1, .6)]


rows, y = [], []
for _ in range(N):
    if rng.random() < DROP_SHARE:
        rows.append(drop("naive" if rng.random() < .6 else "careful")); y.append(1)
    else:
        kind = rng.choice(["regular", "birthday", "seller", "newbie", "own"], p=[.7, .06, .06, .1, .08])
        rows.append(legit(kind)); y.append(0)

X = np.array(rows, dtype=float)
y = np.array(y)
names = [f for f, _ in FEATURES]
np.savetxt("dataset.csv", np.column_stack([X, y]), delimiter=",", fmt="%.3f",
           header=",".join(names + ["is_drop"]), comments="")

# логарифм для «длинных» признаков — так линейной модели проще
LOG = {"card_age_days", "senders_24h", "minutes_to_out", "amount_vs_usual"}
def prep(X):
    X = X.copy()
    for i, n in enumerate(names):
        if n in LOG:
            X[:, i] = np.log1p(X[:, i])
    return X

Xtr, Xte, ytr, yte = train_test_split(prep(X), y, test_size=.3, stratify=y, random_state=0)
mu, sd = Xtr.mean(0), Xtr.std(0)
lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit((Xtr - mu) / sd, ytr)

def report(name, p, thr):
    pred = p >= thr
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    m = dict(auc=round(roc_auc_score(yte, p), 3), recall=round(recall_score(yte, pred), 3),
             precision=round(precision_score(yte, pred), 3), fpr=round(fp / (fp + tn), 4),
             tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
    print(f"{name:10s} порог {thr}: {m}")
    return m

p_lr = lr.predict_proba((Xte - mu) / sd)[:, 1]
THR = 0.8   # удерживаем вывод только при высокой уверенности
m_lr = report("логрег", p_lr, THR)

# простая самопроверка: явный дроп и обычный клиент
def score(v):
    z = (prep(np.array([v], dtype=float)) - mu) / sd
    return lr.predict_proba(z)[0, 1]
assert score([9, 3, 1, 3, 0.95, 1, 17, 12]) > 0.9, "явный дроп должен быть > 90%"
assert score([900, 1, 0, 600, 0.1, 0, 35, 1]) < 0.05, "обычный клиент должен быть < 5%"
assert m_lr["auc"] > 0.9

json.dump(dict(features=FEATURES, log=sorted(LOG), mean=mu.round(5).tolist(), std=sd.round(5).tolist(),
               coef=lr.coef_[0].round(5).tolist(), intercept=round(float(lr.intercept_[0]), 5),
               threshold=THR, n=N, drop_share=DROP_SHARE, test_size=len(yte),
               metrics=m_lr),
          open("model.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("веса:", dict(zip(names, lr.coef_[0].round(2))))
