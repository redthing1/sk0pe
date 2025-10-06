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

then execute code in the dump:
```sh
uv run python ./scripts/w1dump_runner.py -vvv --dump <dump> --backend unicorn --start <addr> --count 50 --trace
```
