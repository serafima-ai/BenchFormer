tasks_implementations_dict = {}


def register_task(name: str = None) -> type:
    def decorate(cls: type, task_name: str = None) -> type:
        global tasks_implementations_dict

        if not task_name:
            task_name = cls.__module__ + '.' + cls.__name__

        if task_name in tasks_implementations_dict:
            print("Task model class {} is already registered and will be overwritten!".format(task_name))

        tasks_implementations_dict[task_name] = cls
        return cls

    return lambda cls: decorate(cls, name)


class TaskModelBuilder(object):
    tasks_dict = tasks_implementations_dict

    @classmethod
    def build(cls, configs):
        try:
            return cls.tasks_dict[configs.task_name](configs)
        except KeyError:
            raise "{} task not implemented!".format(configs.task_name)
