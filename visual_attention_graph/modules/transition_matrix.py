class TransitionMatrix:

    def __init__(self):

        self.transitions = {}

    def update(self, scanpath):

        for i in range(len(scanpath)-1):

            src = scanpath[i]
            dst = scanpath[i+1]

            for s in src:
                for d in dst:

                    key = (s,d)

                    self.transitions[key] = \
                        self.transitions.get(key,0) + 1

    def get_edges(self):

        return self.transitions.items()