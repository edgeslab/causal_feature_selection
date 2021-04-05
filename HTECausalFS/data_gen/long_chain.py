from HTECausalFS.util import *


def continuous_effect(num_examples=1000, num_x=1, variance=1.0, seed=724):
    np.random.seed(seed)

    a = np.random.normal(size=(num_examples, num_x))
    x = np.random.normal(size=(num_examples, num_x))

    # a, x -> t
    at = np.random.uniform(-1, 1, size=num_x)
    xt = np.random.uniform(-1, 1, size=num_x)
    propensity = np.dot(a, at) + np.dot(x, xt) + np.random.normal(scale=variance, size=num_examples)
    propensity = sigmoid(propensity)
    rand = np.random.rand(num_examples)
    t = np.less_equal(rand, propensity).astype(int)

    # t -> b
    tb = np.random.uniform(-1, 1, size=num_x)
    b = tb * t[:, None] + np.random.normal(scale=variance, size=(num_examples, num_x))

    # c parent
    c = np.random.normal(size=(num_examples, num_x))

    # c, t -> d
    cd = np.random.uniform(-1, 1, size=num_x)
    tau = np.random.uniform(-1, 1, size=num_x)
    dnoise = np.random.normal(scale=variance, size=(num_examples, num_x))
    d = cd * c + tau * t[:, None] + dnoise

    # d -> e
    de = np.random.uniform(-1, 1, num_x)
    e = de * d + np.random.normal(scale=variance, size=(num_examples, num_x))

    f = np.random.normal(size=(num_examples, num_x))

    # x, f, d -> Y
    xy = np.random.uniform(-1, 1, size=num_x)
    fy = np.random.uniform(-1, 1, size=num_x)
    dy = np.random.uniform(-1, 1, size=num_x)
    # ----------------------------------------------------------------
    # Interaction between D and X
    # ----------------------------------------------------------------
    interaction_function = np.random.uniform(-1, 1, size=num_x) * x
    dyx = np.sum(d * interaction_function, axis=1)
    y_noise = np.random.normal(scale=variance, size=num_examples)
    y = np.dot(f, fy) + np.dot(x, xy) + np.dot(d, dy) + dyx + y_noise

    # y -> g
    yg = np.random.uniform(-1, 1, size=num_x)
    g = yg * y[:, None] + np.random.normal(scale=variance, size=(num_examples, num_x))

    effect = np.sum(tau * (dy + interaction_function), axis=1)

    dat_dict = dict()
    for j in range(x.shape[1]):
        dat_dict[f"x{j}"] = x[:, j]
        dat_dict[f"a{j}"] = a[:, j]
        dat_dict[f"c{j}"] = c[:, j]
        dat_dict[f"b{j}"] = b[:, j]
        dat_dict[f"f{j}"] = f[:, j]
        dat_dict[f"e{j}"] = e[:, j]
        dat_dict[f"g{j}"] = g[:, j]
        dat_dict[f"d{j}"] = d[:, j]
    dat_dict["t"] = t
    dat_dict["y"] = y
    # dat_dict["a"] = a
    # dat_dict["b"] = b
    # dat_dict["c"] = c
    # dat_dict["d"] = d
    # dat_dict["e"] = e
    # dat_dict["f"] = f
    # dat_dict["g"] = g
    dat_dict["effect"] = effect
    data = pd.DataFrame(dat_dict)

    return data


def step_effect(num_examples=1000, num_x=1, variance=1.0, steps=2, seed=724):
    np.random.seed(seed)

    a = np.random.normal(size=num_examples)
    x = np.random.normal(size=(num_examples, num_x))

    # a, x -> t
    at = np.random.uniform(-1, 1)
    xt = np.random.uniform(-1, 1, size=num_x)
    propensity = at * a + np.dot(x, xt) + np.random.normal(scale=variance, size=num_examples)
    propensity = sigmoid(propensity)
    rand = np.random.rand(num_examples)
    t = np.less_equal(rand, propensity).astype(int)

    # t -> b
    tb = np.random.uniform(-1, 1)
    b = tb * t + np.random.normal(scale=variance, size=num_examples)

    # c parent
    c = np.random.normal(size=num_examples)

    # c, t -> d
    cd = np.random.uniform(-1, 1)
    tau = np.random.uniform(-1, 1)
    dnoise = np.random.normal(scale=variance, size=num_examples)
    d = cd * c + tau * t + dnoise

    # d -> e
    de = np.random.uniform(-1, 1)
    e = de * d + np.random.normal(scale=variance, size=num_examples)

    f = np.random.normal(size=num_examples)

    # x, f, d -> Y
    xy = np.random.uniform(-1, 1, size=num_x)
    fy = np.random.uniform(-1, 1)
    dy = np.random.uniform(-1, 1)
    # ----------------------------------------------------------------
    # Interaction between D and X
    # ----------------------------------------------------------------
    xd = np.random.uniform(-1, 1, size=num_x)
    xd = np.dot(x, xd)
    step_division = np.linspace(np.min(xd), np.max(xd), steps + 1)
    step_effects = np.linspace(-1, 1, steps)
    dyx = np.zeros(num_examples)
    for i, division in enumerate(step_division[:-1]):
        dyx[xd >= division] = step_effects[i]
    y_noise = np.random.normal(scale=variance, size=num_examples)
    y = fy * f + np.dot(x, xy) + dy * d + dyx * d + y_noise

    # y -> g
    yg = np.random.uniform(-1, 1)
    g = yg * y + np.random.normal(scale=variance, size=num_examples)

    effect = tau * (dy + dyx)

    dat_dict = dict()
    for j in range(x.shape[1]):
        dat_dict[f"x{j}"] = x[:, j]
    dat_dict["t"] = t
    dat_dict["y"] = y
    dat_dict["a"] = a
    dat_dict["b"] = b
    dat_dict["c"] = c
    dat_dict["d"] = d
    dat_dict["e"] = e
    dat_dict["f"] = f
    dat_dict["g"] = g
    dat_dict["effect"] = effect
    data = pd.DataFrame(dat_dict)

    return data
