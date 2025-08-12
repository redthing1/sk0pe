# sk0pe

lightweight static binary analysis with emulation (unicorn + triton)

## demo: emulation

a demonstration of emulated function calling in a sample binary:

```sh
uv run python ./scripts/demo_emu_funcs.py -vvv --backend unicorn --trace
uv run python ./scripts/demo_emu_funcs.py -vvv --backend triton --trace
```
