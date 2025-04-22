from metaflow import FlowSpec, step, Parameter

class Counterflow(FlowSpec):
    begin_count = Parameter('ct', default = 20, type = int, required = True) # why is this not self?
    @step
    def start(self):
        self.count = self.begin_count # as long as you assign it to self it saves it as an artifact and functions ahead can identify 
        self.next(self.add)

    @step
    def add(self):
        print("The count is", self.count, "before incrementing")
        self.count += 1
        self.next(self.end)

    @step
    def end(self):
        self.count += 1
        print("Final count is", self.count)

if __name__ == '__main__':
    Counterflow()