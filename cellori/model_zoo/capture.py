from flax import linen as nn


class CaptureModule(nn.Module):

    def capture(self, capture_list, captures, x, name):
        if len(self.variables) > 0:
            if name in capture_list:
                if name in captures:
                    captures[name] = (*captures[name], x)
                else:
                    captures[name] = (x, )
            return captures

    @staticmethod
    def output(capture_list, captures, x):
        if len(capture_list) > 0:
            return x, captures
        else:
            return x
