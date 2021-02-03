def use_nwhy(object_name):
    """
    Replaces the NWHy
    
    Parameters
    ----------
    object_type : nwhy.NWHypergraph or 
        Description
    """
    @decorator
    def _use_nwhy(func, *args, **kwargs):
        this_object = args[0]
        if this_object.nwhy != True:
            return func(*args,**kwargs)
        else:
            args = args[1:]
            object_type = getattr(nwhy, object_name)
            if object_type == nwhy.NWHypergraph:
                g = this_object.state_dict['g']
                return getattr(object_type,func.__name__)(g,*args,**kwargs)
            else:
                s = kwargs['s']
                args = kargs.remove['s']
                lgtype = 'sedgelg' if kwargs[edge]==True else 'snodelg'
                lg = this_object.state_dict[lgtype][s]
                return getattr(object_type,func.__name__)(lg,*args,**kwargs)
