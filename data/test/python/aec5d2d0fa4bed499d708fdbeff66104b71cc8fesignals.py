import django.dispatch

in_access = django.dispatch.Signal(providing_args=["event"])
in_attrib = django.dispatch.Signal(providing_args=["event"])
in_close_nowrite = django.dispatch.Signal(providing_args=["event"])
in_close_write = django.dispatch.Signal(providing_args=["event"])
in_create = django.dispatch.Signal(providing_args=["event"])
in_delete = django.dispatch.Signal(providing_args=["event"])
in_delete_self = django.dispatch.Signal(providing_args=["event"])
in_ignored = django.dispatch.Signal(providing_args=["event"])
in_modify = django.dispatch.Signal(providing_args=["event"])
in_move_self = django.dispatch.Signal(providing_args=["event"])
in_moved_from = django.dispatch.Signal(providing_args=["event"])
in_moved_to = django.dispatch.Signal(providing_args=["event"])
in_open = django.dispatch.Signal(providing_args=["event"])
in_q_overflow = django.dispatch.Signal(providing_args=["event"])
in_unmount = django.dispatch.Signal(providing_args=["event"])
