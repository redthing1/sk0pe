from enum import IntFlag


class MemoryPermissions(IntFlag):
    NONE = 0
    READ = 0b001
    WRITE = 0b010
    EXECUTE = 0b100

    RW = READ | WRITE
    RX = READ | EXECUTE
    RWX = READ | WRITE | EXECUTE

    @classmethod
    def from_rwx(cls, read: bool, write: bool, execute: bool) -> "MemoryPermissions":
        perms = cls.NONE
        if read:
            perms |= cls.READ
        if write:
            perms |= cls.WRITE
        if execute:
            perms |= cls.EXECUTE
        return perms
