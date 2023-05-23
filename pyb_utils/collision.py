from dataclasses import dataclass

import numpy as np
import pybullet as p
import roboticstoolbox
from spatialmath.base import r2q


@dataclass
class NamedCollisionObject:
    """Name of a body and one of its links.

    The body name must correspond to the key in the `bodies` dict, but is
    otherwise arbitrary. The link name should match the URDF. The link name may
    also be None, in which case the base link (index -1) is used.
    """

    body_name: str
    link_name: str = None


@dataclass
class IndexedCollisionObject:
    """Index of a body and one of its links."""

    body_uid: int
    link_uid: int


def index_collision_pairs(physics_uid, bodies, named_collision_pairs):
    """Convert a list of named collision pairs to indexed collision pairs.

    In other words, convert named bodies and links to the indexes used by
    PyBullet to facilate computing collisions between the objects.

    Parameters:
      physics_uid: Index of the PyBullet physics server to use.
      bodies: dict with body name keys and corresponding indices as values
      named_collision_pairs: a list of 2-tuples of NamedCollisionObject

    Returns: a list of 2-tuples of IndexedCollisionObject
    """

    # build a nested dictionary mapping body names to link names to link
    # indices
    body_link_map = {}
    for name, uid in bodies.items():
        body_link_map[name] = {}
        n = p.getNumJoints(uid, physics_uid)
        for i in range(n):
            info = p.getJointInfo(uid, i, physics_uid)
            link_name = info[12].decode("utf-8")
            body_link_map[name][link_name] = i

    def _index_named_collision_object(obj):
        """Map body and link names to corresponding indices."""
        body_uid = bodies[obj.body_name]
        if obj.link_name is not None:
            link_uid = body_link_map[obj.body_name][obj.link_name]
        else:
            link_uid = -1
        return IndexedCollisionObject(body_uid, link_uid)

    # convert all pairs of named collision objects to indices
    indexed_collision_pairs = []
    for a, b in named_collision_pairs:
        a_indexed = _index_named_collision_object(a)
        b_indexed = _index_named_collision_object(b)
        indexed_collision_pairs.append((a_indexed, b_indexed))

    return indexed_collision_pairs


def get_link_ids(physics_uid, uid):
    n = p.getNumJoints(uid, physics_uid)
    body_link_map = {}
    for i in range(n):
        info = p.getJointInfo(uid, i, physics_uid)
        link_name = info[12].decode("utf-8")
        body_link_map[link_name] = i
    return body_link_map


class CollisionDetector:
    def __init__(self, col_id, bodies, named_collision_pairs):
        self.col_id = col_id
        self.robot_id = bodies["robot"]

        self.robot_link_ids = get_link_ids(self.col_id, self.robot_id)
        self.collision_object_ids = [x[1] for x in bodies.items() if x[0] not in ["robot", "dummy_target"]]
        self.indexed_collision_pairs = index_collision_pairs(
            self.col_id, bodies, named_collision_pairs
        )
        self.rtb_model = roboticstoolbox.models.Panda()
        self.link_geometry_collision_dummies = self.get_link_geometry_collision_dummies(self.col_id, self.robot_id)
        self.bodies = bodies

    def compute_distances(self, max_distance=1.0):
        """Compute closest distances for a given configuration.

        Parameters:
          q: Iterable representing the desired configuration. This is applied
             directly to PyBullet body with index bodies["robot"].
          max_distance: Bodies farther apart than this distance are not queried
             by PyBullet, the return value for the distance between such bodies
             will be max_distance.

        Returns: A NumPy array of distances, one per pair of collision objects.
        """

        # put the robot in the given configuration
        # for i in joint_indices:
        #     p.resetJointState(
        #         self.robot_id, i, q[i], physicsClientId=self.col_id
        #     )

        # compute shortest distances between all object pairs
        distances = []
        for a, b in self.indexed_collision_pairs:
            closest_points = p.getClosestPoints(
                a.body_uid,
                b.body_uid,
                distance=max_distance,
                linkIndexA=a.link_uid,
                linkIndexB=b.link_uid,
                physicsClientId=self.col_id,
            )

            # if bodies are above max_distance apart, nothing is returned, so
            # we just saturate at max_distance. Otherwise, take the minimum
            if len(closest_points) == 0:
                distances.append(max_distance)
            else:
                distances.append(np.min([pt[8] for pt in closest_points]))

        return np.array(distances)

    def compute_distances_per_link(self, max_distance=1.0):
        """Compute closest distances configuration sorted by link of robotic arm.

        Parameters:
          q: Iterable representing the desired configuration. This is applied
             directly to PyBullet body with index bodies["robot"].
          max_distance: Bodies farther apart than this distance are not queried
             by PyBullet, the return value for the distance between such bodies
             will be max_distance.

        Returns: A NumPy array of distances, one per pair of collision objects.
        """

        # put the robot in the given configuration
        # for i in joint_indices:
        #     p.resetJointState(
        #         self.robot_id, i, q[i], physicsClientId=self.col_id
        #     )

        # compute shortest distances between all object pairs
        distances = {}
        info = {}
        c_points = {}
        for a, b in self.indexed_collision_pairs:
            closest_points = p.getClosestPoints(
                a.body_uid,
                b.body_uid,
                distance=max_distance,
                linkIndexA=a.link_uid,
                linkIndexB=b.link_uid,
                physicsClientId=self.col_id,
            )

            # visualize shortest distances
            # for abcsd in closest_points:
            #     x = abcsd[5]
            #     y = abcsd[6]
            #     p.addUserDebugLine(x, y, physicsClientId=0)
            # p.removeAllUserDebugItems(physicsClientId=0)

            # if link doesn't have a dict entry, add
            if distances.get(a.link_uid) is None:
                distances[a.link_uid] = []
                c_points[a.link_uid] = []

            # if bodies are above max_distance apart, nothing is returned, so
            # we just saturate at max_distance. Otherwise, take the minimum
            if len(closest_points) == 0:
                distances[a.link_uid].append(max_distance)
            else:
                distances[a.link_uid].append(np.min([pt[8] for pt in closest_points]))
                c = np.min([pt[8] for pt in closest_points])
                idx = [pt[8] for pt in closest_points].index(c)

                x = closest_points[idx][5]
                y = closest_points[idx][6]
                c_points[a.link_uid].append((x,y))

        info["closest_points"] = c_points
        return distances, info

    def compute_distances_per_link_2(self, max_distance=1.0):
        """Compute closest distances configuration sorted by link of robotic arm.

        Parameters:
          q: Iterable representing the desired configuration. This is applied
             directly to PyBullet body with index bodies["robot"].
          max_distance: Bodies farther apart than this distance are not queried
             by PyBullet, the return value for the distance between such bodies
             will be max_distance.

        Returns: A NumPy array of distances, one per pair of collision objects.
        """

        # put the robot in the given configuration
        # for i in joint_indices:
        #     p.resetJointState(
        #         self.robot_id, i, q[i], physicsClientId=self.col_id
        #     )

        # compute shortest distances between all object pairs
        distances = {}
        for a, b in self.indexed_collision_pairs:
            closest_points = p.getClosestPoints(
                a.body_uid,
                b.body_uid,
                distance=max_distance,
                linkIndexA=a.link_uid,
                linkIndexB=b.link_uid,
                physicsClientId=self.col_id,
            )

            # # visualize shortest distances
            # for abcsd in closest_points:
            #     x = abcsd[5]
            #     y = abcsd[6]
            #     p.addUserDebugLine(x, y, physicsClientId=0)
            # p.removeAllUserDebugItems(physicsClientId=0)

            # if link doesn't have a dict entry, add
            if distances.get(a.link_uid) is None:
                distances[a.link_uid] = []

            # if bodies are above max_distance apart, nothing is returned, so
            # we just saturate at max_distance. Otherwise, take the minimum
            if len(closest_points) == 0:
                distances[a.link_uid].append(max_distance)
            else:
                distances[a.link_uid].append([pt[8] for pt in closest_points])

        return distances

    def compute_distance_of_link(self, link_name, obstacle_name, max_distance=1.0):
        """Compute closest distance of a specific link.

        Parameters:
          q: Iterable representing the desired configuration. This is applied
             directly to PyBullet body with index bodies["robot"].
          max_distance: Bodies farther apart than this distance are not queried
             by PyBullet, the return value for the distance between such bodies
             will be max_distance.

        Returns: A NumPy array of distance and points of objects in the world frame
        which connect the line of distance between the objects.
        """

        # todo: reset joint states (see function above)

        # compute shortest distances between all object pairs
        # todo: DEBUG Code PLS DELETE LATER
        # local_frame_pos = []
        # local_frame_orn = []
        # shapes = []
        # csd = p.getCollisionShapeData(objectUniqueId=self.robot_id, linkIndex=self.robot_link_ids[link_name])
        # for c in csd:
        #     local_frame_pos.append(c[5])
        #     local_frame_orn.append(c[6])
        #
        # ls = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.robot_link_ids[link_name])
        # link_world_pos = ls[0]
        # link_world_orn = ls[1]
        #
        # for pos in local_frame_pos:
        #     final_pos = np.array(link_world_pos)+np.array(pos)
        #     p.addUserDebugLine(final_pos, np.array([0,0,0]), np.array([1, 0, 0]), physicsClientId=0)

        result = p.getClosestPoints(
            self.robot_id,
            self.bodies[obstacle_name],
            distance=max_distance,
            linkIndexA=self.robot_link_ids[link_name],
            linkIndexB=-1,
            physicsClientId=self.col_id,
        )

        try:
            return result[0][8], np.array(result[0][5]), np.array(result[0][6])
        except ValueError:
            return None, None, None
        except IndexError:
            # Obstacle is further away than inf_dist
            return None, None, None

    def in_collision(self, q, joint_indices, margin=0.0):
        """Returns True if configuration q is in collision, False otherwise.

        Parameters:
          q: Iterable representing the desired configuration.
          margin: Distance at which objects are considered in collision.
             Default is 0.0.
        """
        ds = self.compute_distances(q, max_distance=margin, joint_indices=joint_indices)
        return (ds < margin).any()

    # def create_dummy_object(self, type):
    #
    #     baseVisualShapeIndex = self.physics_client.createVisualShape(geom_type, **visual_kwargs)
    #     if not ghost:
    #         baseCollisionShapeIndex = self.physics_client.createCollisionShape(geom_type, **collision_kwargs)
    #     else:
    #         baseCollisionShapeIndex = -1
    #     idx = self._bodies_idx[body_name] = self.physics_client.createMultiBody(
    #         baseVisualShapeIndex=baseVisualShapeIndex,
    #         baseCollisionShapeIndex=baseCollisionShapeIndex,
    #         basePosition=,
    #         baseOrientation=,
    #         baseMass=mass,
    #         basePosition=position,
    #     )

    def get_link_geometry_collision_dummies(self, physics_uid, uid):
        links, n, _ = self.rtb_model.get_path(start=self.rtb_model.link_dict["panda_link1"],
                                              end=self.rtb_model.link_dict["panda_hand"])

        link_geometry_ids = {}
        for link in links:
            link_id = self.robot_link_ids[link.name]
            for col in link.collision.data:
                if col.stype == "sphere":
                    id = p.createCollisionShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=col.radius,
                        collisionFramePosition=col.T[:3, -1],
                        collisionFrameOrientation=r2q(col.T[:3, :3],
                                                      order="xyzs"))
                    link_geometry_ids[id] = link_id
                elif col.stype == "cylinder":
                    id = p.createCollisionShape(
                        shapeType=p.GEOM_CYLINDER,
                        radius=col.radius,
                        height=col.length,
                        collisionFramePosition=col.T[:3, -1],
                        collisionFrameOrientation=r2q(col.T[:3, :3],
                                                      order="xyzs"))
                    link_geometry_ids[id] = link_id

                p.setCollisionFilterPair(
                    self.robot_id,
                    id,
                    linkIndexA=link_id,
                    linkIndexB=-1,
                    enableCollision=0,
                    physicsClientId=self.col_id,
                )
        return link_geometry_ids

    def set_collision_geometries(self):

        for link_id in self.robot_link_ids.values():
            collision_data = p.getCollisionShapeData(objectUniqueId=self.robot_id, linkIndex=link_id)
            link_geometry_ids = [key for (key, value) in
                                 self.link_geometry_collision_dummies.items() if value == link_id]

            # iterate trough the corresponding geometries and sync position and orientation with dummy collision geometries
            for geometry_id, col in zip(link_geometry_ids, collision_data):
                pos = col[5]
                orn = col[6]
                p.resetBasePositionAndOrientation(bodyUniqueId=geometry_id, posObj=pos, ornObj=orn)

                # debug
                # deb = p.getBasePositionAndOrientation(bodyUniqueId=geometry_id)
                # print(deb)


def compute_distance(link_col_id, col_id, client_id, max_distance=1.0):
    """Compute closest distance between two geometries

    Parameters:
      q: Iterable representing the desired configuration. This is applied
         directly to PyBullet body with index bodies["robot"].
      max_distance: Bodies farther apart than this distance are not queried
         by PyBullet, the return value for the distance between such bodies
         will be max_distance.

    Returns: A NumPy array of distance and points of objects in the world frame
    which connect the line of distance between the objects.
    """

    # todo: reset joint states (see function above)

    # csd = p.getCollisionShapeData(objectUniqueId = 0, linkIndex=7, physicsClientId=0)
    # p.addUserDebugLine(np.array((0,0,0)), csd[0][5], physicsClientId=0)

    result = p.getClosestPoints(
        link_col_id,
        col_id,
        distance=max_distance,
        physicsClientId=client_id,
    )

    try:
        return result[0][8], np.array(result[0][5]), np.array(result[0][6])
    except ValueError:
        return None, None, None
    except IndexError:
        # Obstacle is further away than inf_dist
        return None, None, None
