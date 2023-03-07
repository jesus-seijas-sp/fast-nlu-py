def measure(net, corpus):
    total = 0
    good = 0
    for item in corpus["data"]:
        for test in item["tests"]:
            output = net.run(test)
            total += 1
            intent = output[0]["intent"] if isinstance(output, list) else output["intent"]
            if intent == item["intent"]:
                good += 1
    return {"good": good, "total": total}
