# import networks

# networks mapping


def get_network(config, net_type):
    if net_type == "SST":
        M = SST
    elif net_type == "BiSST":
        M = BiSST
    else:
        raise NotImplementedError(
            "Not supported proposal network ({})".format(net_type))
    return M(config)

