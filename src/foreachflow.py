from metaflow import FlowSpec, step

class ForeachFlow(FlowSpec):

    @step
    def start(self):
        self.creatures = ['bird', 'mouse', 'dog', 'cockroach']
        self.next(self.analyze_creatures, foreach='creatures') # referring to our list of creatures, create a node (task) for each creature

    @step
    def analyze_creatures(self):
        print("Analyzing", self.input)
        self.creature = self.input
        self.score = len(self.creature)
        self.next(self.join)

    @step 
    def join(self, inputs):
        self.best = max(inputs, key=lambda x: x.score).creature # similar to argmax, this will return which input has maximum score --- change to min to see what happens
        self.next(self.end)

    @step
    def end(self):
        print(self.best, 'won!')

if __name__ == '__main__':
    ForeachFlow()