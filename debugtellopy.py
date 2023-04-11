import inspect
from functools import wraps
from djitellopy import Tello


def dump_args(func):
    """
    Decorator to print function call details.
    This includes parameters names and effective values.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        print(f"{func.__module__}.{func.__qualname__} ( {func_args_str} )")
        return func(*args, **kwargs)

    return wrapper


class DebugTello(Tello):
    def __init__(self):
        super().__init__()

    @dump_args
    def send_rc_control(self, left_right_velocity: int, forward_backward_velocity: int, up_down_velocity: int,
                        yaw_velocity: int):
        pass

    @dump_args
    def takeoff(self):
        self.is_flying = True

    @dump_args
    def land(self):
        self.is_flying = False

    @dump_args
    def move_up(self, x: int):
        pass


if __name__ == '__main__':
    @dump_args
    def test(a, b=4, c="blah-blah", *args, **kwargs):
        pass

    test(1, 2, 3, 4, 5, d=6, g=12.9)
