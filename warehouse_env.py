"""
JSSPLA Environment: Job Shop Scheduling Problem with Limited AMRs in Smart Warehouse
Based on: "Job Shop-Based Scheduling Optimization for Multi-AMR Warehouse Systems
and Validation via Gazebo Simulation" (Supervisor: Dr. Truong Ngoc Cuong, Student: Tran Viet Trung An)

Models:
- 33m x 16.5m warehouse layout with 4 Rack clusters (R1..R4) and dedicated workstations.
- 8 AMRs (2 dedicated AMRs per cluster: R1->[0,1], R2->[2,3], R3->[4,5], R4->[6,7]).
- Inbound & Outbound job flows with delivery tasks and machining operations.
- Battery discharge, charging station detour, loading/unloading times.
- Dual-decision space: Operation Sequence (OS) & AMR Assignment (AA).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
import random
import numpy as np


@dataclass
class Point2D:
    x: float
    y: float

    def distance_to(self, other: "Point2D", metric: str = "manhattan") -> float:
        """Calculates distance between two warehouse points."""
        if metric == "manhattan":
            return abs(self.x - other.x) + abs(self.y - other.y)
        return math.hypot(self.x - other.x, self.y - other.y)


# Warehouse Layout Coordinates (in meters, matching 33m x 16.5m layout in paper)
STATION_COORDS = {
    "BUFFER_1": Point2D(2.0, 10.0),       # Inbound inspection & staging
    "RECEIVING": Point2D(2.0, 13.0),      # Inbound dock
    "RACK_1": Point2D(8.5, 9.0),          # Cluster 1
    "RACK_2": Point2D(15.0, 9.0),         # Cluster 2
    "RACK_3": Point2D(21.5, 9.0),         # Cluster 3
    "RACK_4": Point2D(28.0, 9.0),         # Cluster 4
    "CHARGING": Point2D(2.0, 2.0),        # Automated charging station
    "PICKING": Point2D(8.5, 2.0),         # Picking workstation
    "PACKING": Point2D(15.0, 2.0),        # Packing workstation
    "BUFFER_2": Point2D(21.5, 2.0),       # Outbound staging buffer
    "SHIPPING": Point2D(28.0, 2.0),       # Outbound shipping dock
}

MACHINE_NAMES = ["BUFFER_1", "RACK", "PICKING", "PACKING", "BUFFER_2", "SHIPPING"]
MACHINE_TO_ID = {name: idx for idx, name in enumerate(MACHINE_NAMES)}


@dataclass
class TransportTask:
    """Represents an AMR transport task V_ij moving between stations."""
    job_id: int
    stage: int
    from_station: str
    to_station: str
    from_pos: Point2D
    to_pos: Point2D
    loaded_dist: float
    loaded_time: float


@dataclass
class JobOperation:
    """Represents a stationary machining/handling operation O_ij."""
    job_id: int
    op_idx: int
    workstation_name: str
    workstation_id: int
    duration: float
    transport_before: Optional[TransportTask] = None


@dataclass
class WarehouseJob:
    """Represents an Inbound or Outbound order in the warehouse."""
    job_id: int
    job_type: str  # "INBOUND" or "OUTBOUND"
    cluster_id: int  # 0, 1, 2, 3 (Cluster 1..4)
    operations: List[JobOperation]

    @property
    def num_operations(self) -> int:
        return len(self.operations)


@dataclass
class JSSPLASchedule:
    """Detailed evaluation result of a schedule."""
    makespan: float
    job_completion_times: Dict[int, float]
    amr_completion_times: Dict[int, float]
    machine_utilization: Dict[str, float]
    amr_utilization: Dict[int, float]
    amr_empty_travel: Dict[int, float]
    amr_loaded_travel: Dict[int, float]
    charging_detours: int
    is_valid: bool = True
    validation_msg: str = "Schedule is valid."


class WarehouseInstance:
    """
    Holds problem data for a JSSPLA scenario.
    """
    def __init__(
        self,
        name: str,
        jobs: List[WarehouseJob],
        num_amrs: int = 8,
        amr_speed: float = 1.0,  # 1.0 m/s
        t_load: float = 5.0,     # 5s
        t_unload: float = 5.0,   # 5s
        b_max: float = 100.0,
        b_min: float = 20.0,
        b_target: float = 90.0,
        alpha_e: float = 0.04,   # %/s discharge empty
        alpha_l: float = 0.07,   # %/s discharge loaded
        beta_charge: float = 0.4 # %/s charging rate
    ):
        self.name = name
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_amrs = num_amrs
        self.amr_speed = amr_speed
        self.t_load = t_load
        self.t_unload = t_unload
        self.b_max = b_max
        self.b_min = b_min
        self.b_target = b_target
        self.alpha_e = alpha_e
        self.alpha_l = alpha_l
        self.beta_charge = beta_charge

        # Dedicated sub-fleet mapping: Cluster k (0..3) -> AMRs [2k, 2k+1]
        self.cluster_amrs = {
            k: [2 * k, 2 * k + 1] for k in range(4)
        }

        # Calculate base operation sequence and total operations
        self.base_sequence = []
        for job in self.jobs:
            self.base_sequence.extend([job.job_id] * len(job.operations))
        self.total_operations = len(self.base_sequence)

    @classmethod
    def create_scenario(
        cls,
        scenario_id: int = 1,
        seed: int = 42,
    ) -> "WarehouseInstance":
        """
        Creates one of the 4 benchmark scenarios from the paper:
        Scenario 1: 8 Inbound, 16 Outbound (Total 24 jobs) - Baseline
        Scenario 2: 4 Inbound, 24 Outbound (Total 28 jobs) - Downstream bottleneck
        Scenario 3: 20 Inbound, 8 Outbound (Total 28 jobs) - Inbound surge
        Scenario 4: 150 Inbound, 200 Outbound (Total 350 jobs) - Large stress test
        """
        config = {
            1: (8, 16, "Scenario_1_Baseline_24J"),
            2: (4, 24, "Scenario_2_Bottleneck_28J"),
            3: (20, 8, "Scenario_3_Surge_28J"),
            4: (150, 200, "Scenario_4_StressTest_350J"),
        }
        n_in, n_out, name = config.get(scenario_id, (8, 16, "Custom_Scenario"))
        return cls.generate(num_inbound=n_in, num_outbound=n_out, name=name, seed=seed)

    @classmethod
    def generate(
        cls,
        num_inbound: int,
        num_outbound: int,
        name: str = "Warehouse_Scenario",
        seed: Optional[int] = None,
        amr_speed: float = 1.0,
    ) -> "WarehouseInstance":
        """
        Generates warehouse jobs according to real layout and operational flows.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        jobs: List[WarehouseJob] = []
        job_id = 0

        # Helper to compute travel time
        def get_travel(p1: Point2D, p2: Point2D) -> Tuple[float, float]:
            dist = p1.distance_to(p2, metric="manhattan")
            time = dist / amr_speed
            return dist, time

        # 1. Inbound Jobs (Buffer 1 -> Rack Cluster R_k)
        for _ in range(num_inbound):
            cluster_id = random.randint(0, 3)
            rack_name = f"RACK_{cluster_id + 1}"
            b1_pos = STATION_COORDS["BUFFER_1"]
            rack_pos = STATION_COORDS[rack_name]
            dist, t_travel = get_travel(b1_pos, rack_pos)

            trans = TransportTask(
                job_id=job_id,
                stage=0,
                from_station="BUFFER_1",
                to_station=rack_name,
                from_pos=b1_pos,
                to_pos=rack_pos,
                loaded_dist=dist,
                loaded_time=t_travel,
            )

            # Inbound put-away operation at Rack cluster
            dur_putaway = random.uniform(15.0, 35.0)
            op = JobOperation(
                job_id=job_id,
                op_idx=0,
                workstation_name=rack_name,
                workstation_id=MACHINE_TO_ID["RACK"],
                duration=dur_putaway,
                transport_before=trans,
            )

            jobs.append(WarehouseJob(
                job_id=job_id,
                job_type="INBOUND",
                cluster_id=cluster_id,
                operations=[op]
            ))
            job_id += 1

        # 2. Outbound Jobs (Rack -> Picking -> Packing -> Shipping)
        for _ in range(num_outbound):
            cluster_id = random.randint(0, 3)
            rack_name = f"RACK_{cluster_id + 1}"
            rack_pos = STATION_COORDS[rack_name]
            pick_pos = STATION_COORDS["PICKING"]
            pack_pos = STATION_COORDS["PACKING"]
            ship_pos = STATION_COORDS["SHIPPING"]

            # Stage 0: Rack -> Picking
            d0, t0 = get_travel(rack_pos, pick_pos)
            trans0 = TransportTask(
                job_id=job_id, stage=0, from_station=rack_name, to_station="PICKING",
                from_pos=rack_pos, to_pos=pick_pos, loaded_dist=d0, loaded_time=t0
            )
            dur_pick = random.uniform(20.0, 45.0)
            op0 = JobOperation(
                job_id=job_id, op_idx=0, workstation_name="PICKING",
                workstation_id=MACHINE_TO_ID["PICKING"], duration=dur_pick, transport_before=trans0
            )

            # Stage 1: Picking -> Packing
            d1, t1 = get_travel(pick_pos, pack_pos)
            trans1 = TransportTask(
                job_id=job_id, stage=1, from_station="PICKING", to_station="PACKING",
                from_pos=pick_pos, to_pos=pack_pos, loaded_dist=d1, loaded_time=t1
            )
            dur_pack = random.uniform(15.0, 35.0)
            op1 = JobOperation(
                job_id=job_id, op_idx=1, workstation_name="PACKING",
                workstation_id=MACHINE_TO_ID["PACKING"], duration=dur_pack, transport_before=trans1
            )

            # Stage 2: Packing -> Shipping
            d2, t2 = get_travel(pack_pos, ship_pos)
            trans2 = TransportTask(
                job_id=job_id, stage=2, from_station="PACKING", to_station="SHIPPING",
                from_pos=pack_pos, to_pos=ship_pos, loaded_dist=d2, loaded_time=t2
            )
            dur_ship = random.uniform(10.0, 25.0)
            op2 = JobOperation(
                job_id=job_id, op_idx=2, workstation_name="SHIPPING",
                workstation_id=MACHINE_TO_ID["SHIPPING"], duration=dur_ship, transport_before=trans2
            )

            jobs.append(WarehouseJob(
                job_id=job_id,
                job_type="OUTBOUND",
                cluster_id=cluster_id,
                operations=[op0, op1, op2]
            ))
            job_id += 1

        return cls(name=name, jobs=jobs, amr_speed=amr_speed)


class JSSPLADecoder:
    """
    Cooperative Decoder for JSSPLA:
    Simulates execution of:
    - OS (Operation Sequence): list of Job IDs indicating production priority.
    - AA (AMR Assignment): integer index (0 or 1) indicating which of the 2 dedicated AMRs
      serves each transport task.
    """
    def __init__(self, instance: WarehouseInstance):
        self.instance = instance

    def decode(
        self,
        os_seq: List[int],
        aa_seq: List[int],
    ) -> JSSPLASchedule:
        """
        Decodes (OS, AA) into a feasible timeline according to equations (1)-(16) in paper.
        """
        inst = self.instance
        num_jobs = inst.num_jobs
        num_amrs = inst.num_amrs

        # Tracking state
        job_op_idx = [0] * num_jobs
        job_avail_time = [0.0] * num_jobs

        # Machine tracking: machine_id -> available time
        mach_avail_time = {m_id: 0.0 for m_id in MACHINE_TO_ID.values()}
        mach_busy_time = {m_id: 0.0 for m_id in MACHINE_TO_ID.values()}

        # AMR tracking: amr_id -> current time, current pos, current battery
        amr_avail_time = [0.0] * num_amrs
        charging_pos = STATION_COORDS["CHARGING"]
        amr_pos = [Point2D(charging_pos.x, charging_pos.y) for _ in range(num_amrs)]
        amr_battery = [inst.b_max] * num_amrs
        amr_empty_dist = [0.0] * num_amrs
        amr_loaded_dist = [0.0] * num_amrs
        charging_detours = 0

        # Map AMR to cluster
        # Cluster k has AMRs [2k, 2k+1]
        cluster_of_amr = {r: r // 2 for r in range(num_amrs)}

        for k, job_id in enumerate(os_seq):
            job = inst.jobs[job_id]
            op_idx = job_op_idx[job_id]
            job_op_idx[job_id] += 1
            op = job.operations[op_idx]

            # 1. Determine assigned AMR
            # AA designates 0 or 1 for the 2 dedicated AMRs of this job's cluster
            cluster_id = job.cluster_id
            dedicated_amrs = inst.cluster_amrs[cluster_id]
            amr_choice = aa_seq[k] % len(dedicated_amrs) if k < len(aa_seq) else 0
            amr_id = dedicated_amrs[amr_choice]

            trans = op.transport_before
            start_delivery = job_avail_time[job_id]

            if trans is not None:
                # Calculate empty travel from AMR current location to pickup point
                curr_amr_pos = amr_pos[amr_id]
                pickup_pos = trans.from_pos
                dropoff_pos = trans.to_pos

                empty_d = curr_amr_pos.distance_to(pickup_pos, metric="manhattan")
                empty_t = empty_d / inst.amr_speed
                loaded_d = trans.loaded_dist
                loaded_t = trans.loaded_time

                # Check battery: estimated battery cost
                batt_needed = (empty_t * inst.alpha_e) + (loaded_t * inst.alpha_l)
                curr_batt = amr_battery[amr_id]

                time_amr_ready = amr_avail_time[amr_id]

                # Check if charging detour is needed (Equation 13a, 13b)
                if curr_batt - batt_needed < inst.b_min:
                    # Detour to charging station
                    charging_detours += 1
                    d_to_chg = curr_amr_pos.distance_to(charging_pos, metric="manhattan")
                    t_to_chg = d_to_chg / inst.amr_speed
                    batt_after_reach = curr_batt - (t_to_chg * inst.alpha_e)
                    charge_needed = max(0.0, inst.b_target - batt_after_reach)
                    t_charge = charge_needed / inst.beta_charge

                    time_amr_ready += t_to_chg + t_charge
                    curr_amr_pos = Point2D(charging_pos.x, charging_pos.y)
                    curr_batt = inst.b_target

                    # Recalculate empty travel from charging station to pickup
                    empty_d = curr_amr_pos.distance_to(pickup_pos, metric="manhattan")
                    empty_t = empty_d / inst.amr_speed

                # AMR arrives at pickup point at:
                amr_reach_pickup = time_amr_ready + empty_t
                actual_pickup_start = max(amr_reach_pickup, start_delivery)

                # Delivery finishes after loading, loaded travel, and unloading (Constraint 8)
                delivery_finish = actual_pickup_start + inst.t_load + loaded_t + inst.t_unload

                # Update AMR state
                amr_battery[amr_id] = curr_batt - (empty_t * inst.alpha_e) - (loaded_t * inst.alpha_l)
                amr_pos[amr_id] = Point2D(dropoff_pos.x, dropoff_pos.y)
                amr_avail_time[amr_id] = delivery_finish
                amr_empty_dist[amr_id] += empty_d
                amr_loaded_dist[amr_id] += loaded_d

                job_ready_for_op = delivery_finish
            else:
                job_ready_for_op = start_delivery

            # 2. Schedule Machine Operation
            m_id = op.workstation_id
            m_ready = mach_avail_time[m_id]
            op_start = max(job_ready_for_op, m_ready)
            op_finish = op_start + op.duration

            # Update machine state
            mach_avail_time[m_id] = op_finish
            mach_busy_time[m_id] += op.duration

            # Update job state
            job_avail_time[job_id] = op_finish

        makespan = max(max(job_avail_time), max(amr_avail_time))

        # Metrics
        mach_util = {}
        for m_name, m_id in MACHINE_TO_ID.items():
            busy = mach_busy_time[m_id]
            mach_util[m_name] = (busy / makespan * 100.0) if makespan > 0 else 0.0

        amr_util = {}
        for r in range(num_amrs):
            active_t = amr_avail_time[r]
            amr_util[r] = (active_t / makespan * 100.0) if makespan > 0 else 0.0

        return JSSPLASchedule(
            makespan=makespan,
            job_completion_times={j: job_avail_time[j] for j in range(num_jobs)},
            amr_completion_times={r: amr_avail_time[r] for r in range(num_amrs)},
            machine_utilization=mach_util,
            amr_utilization=amr_util,
            amr_empty_travel={r: amr_empty_dist[r] for r in range(num_amrs)},
            amr_loaded_travel={r: amr_loaded_dist[r] for r in range(num_amrs)},
            charging_detours=charging_detours,
            is_valid=True,
        )
