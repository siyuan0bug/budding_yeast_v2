"""Map production knobs to explicit constructor arguments, never ignored **kwargs."""
import inspect


def constructor_kwargs(model_class, *, num_vars, param_dim, width, n_blocks, version=1):
    params = inspect.signature(model_class.__init__).parameters
    if version == 0:
        # Exact historical generic branch, for old checkpoint shapes only.
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        return {key: value for key, value in {'num_vars': num_vars, 'param_dim': param_dim, 'width': width}.items()
                if key in params or accepts_kwargs}
    if version != 1:
        raise ValueError(f'Unknown baseline constructor contract: {version}')
    result = {key: value for key, value in (('num_vars', num_vars), ('param_dim', param_dim)) if key in params}
    for aliases, value in ((('width', 'd_model', 'hidden_dim', 'embed_dim'), width),
                           (('n_blocks', 'e_layers', 'n_layers', 'num_layers', 'depth'), n_blocks)):
        for name in aliases:
            if name in params:
                result[name] = value
                break
    return result
