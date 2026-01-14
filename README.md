# sk0pe

lightweight static binary analysis with emulation (unicorn + triton)

## demo: emulation

a demonstration of emulated function calling in a sample binary:

```sh
uv run python ./scripts/demo_emu_funcs.py -vvv --backend unicorn --trace
uv run python ./scripts/demo_emu_funcs.py -vvv --backend triton --trace
```

## demo: run a `w1dump`

create a `w1dump` using [w1tn3ss](https://github.com/redthing1/w1tn3ss)

then symbolically execute code in the dump:

```sh
uv run ./scripts/w1dump_symbolic_emu.py -vvv --backend triton --count 40 --symbolize-all-gpr <dump> --start <addr> --trace
```
