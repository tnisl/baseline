"""Output contracts for guided JSON decoding; tool values still come from the model."""
def object_schema(properties,required):
    return {"type":"object","properties":properties,"required":required,"additionalProperties":False}


def role_schema(role):
    thought={"type":"string","maxLength":320}
    tool={"type":"string"}
    action=object_schema({"name":tool,"arguments":{"type":"object"}},["name","arguments"])
    answer={"type":"string","maxLength":1024}
    if role=="caller":
        return action
    if role=="summarizer":
        return object_schema({"answer":answer},["answer"])
    if role=="planner":
        return object_schema({"thought":thought,"next":{"type":"string","enum":["caller","summarizer","give_up"]},
                              "tool":tool},["thought","next"])
    if role=="global":
        return {"anyOf":[
            object_schema({"thought":thought,"next":{"type":"string","enum":["caller"]},
                           "tool":tool,"action":action},["thought","next","tool","action"]),
            object_schema({"thought":thought,"next":{"type":"string","enum":["summarizer"]},
                           "answer":answer},["thought","next","answer"]),
            object_schema({"thought":thought,"next":{"type":"string","enum":["give_up"]},
                           "answer":answer},["thought","next"])
        ]}
    raise ValueError("Unknown role: "+role)
