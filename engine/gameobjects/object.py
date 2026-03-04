class GameObject:
    def __init__(self, transform, mesh=None, material=None, collider=None, obj_name=None):
        """
        Docstring für __init__

        :param self: The object itself
        :param mesh: The mesh of the object
        :param material: The material of the object
        :param collider: The collider of the object
        :param obj_name: The name of the object

        modualar:
        -> components: list of components (e.g. physics, audio, light)
        """
        self.transform = transform
        self.mesh = mesh
        self.material = material
        self.collider = collider
        self.obj_name = obj_name or "GameObject(name_placeholder)"
        
        self.components = []
        
    def add_component(self, component):
        component.game_object = self  # Set back-reference
        self.components.append(component)
        
        if hasattr(component, "start"):
            component.start()
            
    def get_component(self, component_type):
        for comp in self.components:
            if isinstance(comp, component_type):
                return comp
        return None
    
    def update(self, delta_time):
        for comp in self.components:
            if hasattr(comp, "update"):
                comp.update(delta_time)