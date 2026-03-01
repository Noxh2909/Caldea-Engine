class GameObject:
    def __init__(self, mesh, transform, material=None, collider=None, light=None, audio_config=None):
        """
        Docstring für __init__

        :param self: The object itself
        :param mesh: The mesh of the object
        :param transform: The transform of the object
        :param material: The material of the object
        :param collider: The collider of the object
        :param light: The light of the object
        :param audio_config: The audio configuration of the object
        """
        self.mesh = mesh
        self.transform = transform
        self.material = material
        self.collider = collider
        self.light = light
        self.audio_config = audio_config
