import json

edges = []

def conn(ls, rs, dep):
    for i in ls:
        for j in rs:
            edges.append({"s": i, "t": j, "w": 2**(dep-1)})

def dfs(depot, dep):
    if dep == 0:
        return {depot}, 1

    lc, lsiz = dfs(depot, dep - 1)
    rc, rsiz = dfs(depot + lsiz, dep - 1)

    mid = {depot + lsiz + rsiz}

    conn(lc, rc, dep)
    conn(rc, mid, dep)
    conn(mid, lc, dep)

    return lc | rc, lsiz + rsiz + 1

depth = 4
allc, allsiz = dfs(1, depth)
conn({0}, allc, depth)
conn(allc, {0}, depth)
n = allsiz + 1  # +1 for depot
prob = [1 if i in allc else 0.5 for i in range(n)]
prob[0] = 1  # depot always active

data = {
    "n": n,
    "sym": False,
    "edges": edges,
    "prob": prob,
}
print(json.dumps(data, indent=2))
