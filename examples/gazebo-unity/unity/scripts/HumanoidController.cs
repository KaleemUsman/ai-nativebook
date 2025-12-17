using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

[RequireComponent(typeof(ArticulationBody))]
public class HumanoidController : MonoBehaviour
{
    [System.Serializable]
    public class JointMapping
    {
        public string jointName;
        public ArticulationBody jointArticulationBody;
        public JointType jointType;
        public float minAngle;
        public float maxAngle;
        public float maxForce = 100f;
        public float maxTorque = 100f;
    }

    public enum JointType
    {
        Revolute,
        Prismatic,
        Fixed
    }

    [Header("Joint Configuration")]
    public JointMapping[] jointMappings;

    [Header("ROS Communication")]
    public string robotNamespace = "/humanoid";
    public string jointStateTopic = "/joint_states";
    public string cmdTopic = "/joint_commands";

    [Header("Control Parameters")]
    public float interpolationSpeed = 10f;
    public bool useROSControl = true;

    private ROSConnection ros;
    private Dictionary<string, float> targetJointPositions;
    private Dictionary<string, float> currentJointPositions;

    // Animation and movement parameters
    [Header("Animation Parameters")]
    public float walkSpeed = 1.0f;
    public float turnSpeed = 50.0f;
    public float stepHeight = 0.1f;
    public float stepFrequency = 2.0f;

    private float animationTimer = 0f;
    private bool isWalking = false;
    private Vector3 movementDirection = Vector3.zero;
    private Vector3 targetPosition = Vector3.zero;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.instance;

        // Initialize dictionaries
        targetJointPositions = new Dictionary<string, float>();
        currentJointPositions = new Dictionary<string, float>();

        // Initialize joint positions
        foreach (var joint in jointMappings)
        {
            if (joint.jointArticulationBody != null)
            {
                float initialPosition = GetJointPosition(joint);
                targetJointPositions[joint.jointName] = initialPosition;
                currentJointPositions[joint.jointName] = initialPosition;
            }
        }

        // Subscribe to ROS topics if using ROS control
        if (useROSControl)
        {
            ros.Subscribe<sensor_msgs.JointStateMsg>(jointStateTopic, OnJointStateReceived);
        }

        // Initialize animation parameters
        targetPosition = transform.position;
    }

    void Update()
    {
        // Update joint positions based on control method
        if (useROSControl)
        {
            UpdateJointsFromROS();
        }
        else
        {
            UpdateJointsManually();
        }

        // Handle animation/movement if applicable
        UpdateAnimation();
    }

    void UpdateJointsFromROS()
    {
        // Interpolate to target joint positions
        foreach (var joint in jointMappings)
        {
            if (targetJointPositions.ContainsKey(joint.jointName))
            {
                float currentPosition = currentJointPositions[joint.jointName];
                float targetPosition = targetJointPositions[joint.jointName];

                float newPosition = Mathf.Lerp(currentPosition, targetPosition,
                                              Time.deltaTime * interpolationSpeed);

                SetJointPosition(joint, newPosition);
                currentJointPositions[joint.jointName] = newPosition;
            }
        }
    }

    void UpdateJointsManually()
    {
        // For manual control, you might implement keyboard/gamepad input
        // This is a placeholder for manual control logic
        foreach (var joint in jointMappings)
        {
            if (joint.jointArticulationBody != null)
            {
                // Example: Move joint based on some input
                // float input = Input.GetAxis("Joint_" + joint.jointName);
                // float newPosition = currentJointPositions[joint.jointName] + input * Time.deltaTime;
                // newPosition = Mathf.Clamp(newPosition, joint.minAngle, joint.maxAngle);
                // SetJointPosition(joint, newPosition);
                // currentJointPositions[joint.jointName] = newPosition;
            }
        }
    }

    void UpdateAnimation()
    {
        // Simple walking animation
        if (isWalking)
        {
            animationTimer += Time.deltaTime;

            // Calculate step animation based on timer
            float stepPhase = Mathf.Sin(animationTimer * stepFrequency * 2 * Mathf.PI);
            float stepOffset = stepPhase * stepHeight / 2;

            // Apply simple walking motion
            transform.position = Vector3.Lerp(transform.position, targetPosition,
                                            Time.deltaTime * walkSpeed);

            // Add simple up/down motion for walking
            Vector3 pos = transform.position;
            pos.y += stepOffset;
            transform.position = pos;
        }
    }

    void OnJointStateReceived(sensor_msgs.JointStateMsg jointState)
    {
        // Update target joint positions from ROS message
        for (int i = 0; i < jointState.name.Array.Length; i++)
        {
            string jointName = jointState.name.Array[i];
            float jointPosition = (float)jointState.position[i];

            if (targetJointPositions.ContainsKey(jointName))
            {
                // Clamp to joint limits
                var jointMapping = System.Array.Find(jointMappings, j => j.jointName == jointName);
                if (jointMapping != null)
                {
                    jointPosition = Mathf.Clamp(jointPosition, jointMapping.minAngle, jointMapping.maxAngle);
                }

                targetJointPositions[jointName] = jointPosition;
            }
        }
    }

    float GetJointPosition(JointMapping joint)
    {
        if (joint.jointArticulationBody != null)
        {
            switch (joint.jointType)
            {
                case JointType.Revolute:
                    return joint.jointArticulationBody.jointPosition[0];
                case JointType.Prismatic:
                    return joint.jointArticulationBody.jointPosition[0];
                default:
                    return 0f;
            }
        }
        return 0f;
    }

    void SetJointPosition(JointMapping joint, float position)
    {
        if (joint.jointArticulationBody != null)
        {
            ArticulationDrive drive = joint.jointArticulationBody.xDrive;

            switch (joint.jointType)
            {
                case JointType.Revolute:
                    drive.target = position;
                    drive.forceLimit = joint.maxForce;
                    joint.jointArticulationBody.xDrive = drive;
                    break;
                case JointType.Prismatic:
                    drive.target = position;
                    drive.forceLimit = joint.maxForce;
                    joint.jointArticulationBody.xDrive = drive;
                    break;
            }
        }
    }

    // Public methods for external control
    public void SetJointTarget(string jointName, float targetPosition)
    {
        if (targetJointPositions.ContainsKey(jointName))
        {
            var jointMapping = System.Array.Find(jointMappings, j => j.jointName == jointName);
            if (jointMapping != null)
            {
                targetPosition = Mathf.Clamp(targetPosition, jointMapping.minAngle, jointMapping.maxAngle);
            }
            targetJointPositions[jointName] = targetPosition;
        }
    }

    public void SetJointTargets(Dictionary<string, float> jointTargets)
    {
        foreach (var target in jointTargets)
        {
            SetJointTarget(target.Key, target.Value);
        }
    }

    public void StartWalking(Vector3 direction)
    {
        isWalking = true;
        movementDirection = direction.normalized;
        targetPosition = transform.position + movementDirection;
    }

    public void StopWalking()
    {
        isWalking = false;
    }

    public void SetWalkingSpeed(float speed)
    {
        walkSpeed = speed;
    }

    // Method to send joint states back to ROS (if needed)
    public void PublishJointStates()
    {
        if (!useROSControl) return;

        // Create joint state message
        var jointStateMsg = new sensor_msgs.JointStateMsg();
        jointStateMsg.header = new std_msgs.HeaderMsg();
        jointStateMsg.header.stamp = new builtin_interfaces.TimeMsg(ROSConnection.GetNodeTime());
        jointStateMsg.header.frame_id = "base_link";

        // Set up arrays
        List<string> jointNames = new List<string>();
        List<double> positions = new List<double>();
        List<double> velocities = new List<double>();
        List<double> efforts = new List<double>();

        foreach (var joint in jointMappings)
        {
            jointNames.Add(joint.jointName);
            positions.Add(currentJointPositions[joint.jointName]);
            velocities.Add(0.0); // Placeholder - calculate actual velocity
            efforts.Add(0.0);    // Placeholder - calculate actual effort
        }

        jointStateMsg.name = new StringMsg(jointNames.ToArray());
        jointStateMsg.position = positions.ToArray();
        jointStateMsg.velocity = velocities.ToArray();
        jointStateMsg.effort = efforts.ToArray();

        // Publish the message
        ros.Publish(jointStateTopic, jointStateMsg);
    }

    // Method to send IMU data (if the robot has IMU)
    public void PublishImuData()
    {
        // Implementation would depend on whether the robot has IMU sensors
        // This is a placeholder for potential IMU data publishing
    }

    // Debug visualization
    void OnDrawGizmosSelected()
    {
        if (jointMappings != null)
        {
            foreach (var joint in jointMappings)
            {
                if (joint.jointArticulationBody != null)
                {
                    Gizmos.color = Color.yellow;
                    Gizmos.DrawWireSphere(joint.jointArticulationBody.transform.position, 0.05f);

                    // Draw joint limits visualization
                    if (joint.jointType == JointType.Revolute)
                    {
                        Vector3 forward = joint.jointArticulationBody.transform.forward;
                        Gizmos.color = Color.red;
                        Gizmos.DrawLine(joint.jointArticulationBody.transform.position,
                                      joint.jointArticulationBody.transform.position +
                                      Quaternion.AngleAxis(joint.minAngle, joint.jointArticulationBody.transform.up) * forward * 0.1f);
                        Gizmos.color = Color.green;
                        Gizmos.DrawLine(joint.jointArticulationBody.transform.position,
                                      joint.jointArticulationBody.transform.position +
                                      Quaternion.AngleAxis(joint.maxAngle, joint.jointArticulationBody.transform.up) * forward * 0.1f);
                    }
                }
            }
        }
    }
}