def trie_operations(words, queries):
    # words: list of strings to insert
    # queries: list of (op, value) tuples where op is 'search' or 'startsWith'
    # return: list of booleans, one per query
    root = {}
    END = "#"

    for word in words:
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node[END] = True

    results = []

    for op, value in queries:
        node =  root
        found = True
        for char in value:
            if char not in node:
                found = False
                break
            node = node[char]
        
        if op == "search":
            results.append(found and END in node)
        else:
            results.append(found)

    return results