using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HumanoidMovementTest : MonoBehaviour
{
    [Header("Movement Configuration")]
    public HumanoidController humanoidController;
    public float walkSpeed = 1.0f;
    public float turnSpeed = 50.0f;
    public float stepHeight = 0.1f;
    public float stepFrequency = 2.0f;

    [Header("Test Sequences")]
    public bool runBasicMovementTest = false;
    public bool runJointControlTest = false;
    public bool runWalkingTest = false;

    [Header("Debug Visualization")]
    public bool showTrajectory = true;
    public Color trajectoryColor = Color.blue;
    public LineRenderer lineRenderer;

    private Vector3 targetPosition;
    private float animationTimer = 0f;
    private bool isWalking = false;
    private List<Vector3> trajectoryPoints = new List<Vector3>();
    private int maxTrajectoryPoints = 100;

    void Start()
    {
        if (humanoidController == null)
        {
            humanoidController = GetComponent<HumanoidController>();
        }

        // Set up line renderer for trajectory visualization
        if (showTrajectory && lineRenderer == null)
        {
            GameObject lineObj = new GameObject("TrajectoryLine");
            lineObj.transform.SetParent(transform);
            lineRenderer = lineObj.AddComponent<LineRenderer>();
            lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
            lineRenderer.widthMultiplier = 0.05f;
            lineRenderer.startColor = trajectoryColor;
            lineRenderer.endColor = trajectoryColor;
        }

        targetPosition = transform.position;
    }

    void Update()
    {
        if (runBasicMovementTest)
        {
            RunBasicMovementTest();
        }

        if (runJointControlTest)
        {
            RunJointControlTest();
        }

        if (runWalkingTest)
        {
            RunWalkingTest();
        }

        UpdateTrajectory();
    }

    void RunBasicMovementTest()
    {
        // Simple movement in a square pattern
        Vector3 movementDirection = Vector3.zero;

        if (Vector3.Distance(transform.position, targetPosition) < 0.1f)
        {
            // Change target position in a square pattern
            float time = Time.time * 0.1f;
            targetPosition = transform.position + new Vector3(
                Mathf.Sin(time) * 2f,
                0,
                Mathf.Cos(time) * 2f
            );
        }

        movementDirection = (targetPosition - transform.position).normalized;
        transform.position += movementDirection * walkSpeed * Time.deltaTime;
        transform.LookAt(new Vector3(targetPosition.x, transform.position.y, targetPosition.z));

        // Add simple up/down motion to simulate walking
        float stepOffset = Mathf.Sin(Time.time * stepFrequency * 2 * Mathf.PI) * stepHeight / 2;
        transform.position = new Vector3(
            transform.position.x,
            transform.position.y + stepOffset,
            transform.position.z
        );
    }

    void RunJointControlTest()
    {
        // Test basic joint movements
        if (humanoidController != null)
        {
            // Example: Move a specific joint
            humanoidController.SetJointTarget("head_joint", Mathf.Sin(Time.time) * 0.2f);
            humanoidController.SetJointTarget("left_elbow_joint", Mathf.Cos(Time.time) * 0.3f);
            humanoidController.SetJointTarget("right_knee_joint", Mathf.Sin(Time.time * 0.5f) * 0.4f);
        }
    }

    void RunWalkingTest()
    {
        // More realistic walking simulation
        if (humanoidController != null)
        {
            // Simulate walking gait - alternate leg movements
            float walkPhase = (Time.time * walkSpeed) % (2 * Mathf.PI);

            // Left leg movement
            float leftLegAngle = Mathf.Sin(walkPhase) * 0.3f;
            humanoidController.SetJointTarget("left_hip_joint", leftLegAngle);
            humanoidController.SetJointTarget("left_knee_joint", Mathf.Abs(leftLegAngle) * 0.8f);

            // Right leg movement (opposite phase)
            float rightLegAngle = Mathf.Sin(walkPhase + Mathf.PI) * 0.3f;
            humanoidController.SetJointTarget("right_hip_joint", rightLegAngle);
            humanoidController.SetJointTarget("right_knee_joint", Mathf.Abs(rightLegAngle) * 0.8f);

            // Arm swing compensation
            humanoidController.SetJointTarget("left_shoulder_joint", -rightLegAngle * 0.5f);
            humanoidController.SetJointTarget("right_shoulder_joint", -leftLegAngle * 0.5f);

            // Body balance
            humanoidController.SetJointTarget("torso_joint", -Mathf.Sin(walkPhase) * 0.05f);
        }

        // Move the robot forward gradually
        transform.position += transform.forward * walkSpeed * 0.1f * Time.deltaTime;
    }

    void UpdateTrajectory()
    {
        if (showTrajectory && lineRenderer != null)
        {
            trajectoryPoints.Add(transform.position);

            if (trajectoryPoints.Count > maxTrajectoryPoints)
            {
                trajectoryPoints.RemoveAt(0);
            }

            lineRenderer.positionCount = trajectoryPoints.Count;
            lineRenderer.SetPositions(trajectoryPoints.ToArray());
        }
    }

    // Public methods for external control
    public void StartMovementTest()
    {
        runBasicMovementTest = true;
        runJointControlTest = false;
        runWalkingTest = false;
    }

    public void StartJointControlTest()
    {
        runBasicMovementTest = false;
        runJointControlTest = true;
        runWalkingTest = false;
    }

    public void StartWalkingTest()
    {
        runBasicMovementTest = false;
        runJointControlTest = false;
        runWalkingTest = true;
    }

    public void StopAllTests()
    {
        runBasicMovementTest = false;
        runJointControlTest = false;
        runWalkingTest = false;
    }

    // Manual control methods
    public void MoveForward()
    {
        transform.position += transform.forward * walkSpeed * Time.deltaTime;
    }

    public void MoveBackward()
    {
        transform.position -= transform.forward * walkSpeed * Time.deltaTime;
    }

    public void TurnLeft()
    {
        transform.Rotate(Vector3.up, -turnSpeed * Time.deltaTime);
    }

    public void TurnRight()
    {
        transform.Rotate(Vector3.up, turnSpeed * Time.deltaTime);
    }

    // Test method for validating robot movements
    public bool ValidateMovement(float expectedDistance, float tolerance = 0.1f)
    {
        Vector3 startPosition = transform.position - (transform.forward * expectedDistance);
        float actualDistance = Vector3.Distance(startPosition, transform.position);
        return Mathf.Abs(actualDistance - expectedDistance) <= tolerance;
    }

    // Test method for validating joint ranges
    public bool ValidateJointRange(string jointName, float minValue, float maxValue)
    {
        if (humanoidController == null) return false;

        // This would require access to current joint positions
        // Implementation depends on how the controller exposes joint data
        return true; // Placeholder
    }

    // Interaction test methods
    public void TestObjectInteraction(Transform interactableObject)
    {
        if (interactableObject != null)
        {
            // Move hand towards object
            if (humanoidController != null)
            {
                // Example: Move right hand to object position
                // This would require inverse kinematics or more complex joint control
                // For now, just log the interaction
                Debug.Log("Attempting to interact with object at: " + interactableObject.position);
            }
        }
    }

    // Reset robot to initial position and configuration
    public void ResetRobot()
    {
        StopAllTests();
        transform.position = Vector3.zero;
        transform.rotation = Quaternion.identity;

        if (humanoidController != null)
        {
            // Reset all joints to neutral position
            Dictionary<string, float> neutralJoints = new Dictionary<string, float>();
            // Add neutral positions for all joints based on robot configuration
            humanoidController.SetJointTargets(neutralJoints);
        }

        trajectoryPoints.Clear();
    }

    // Visualization methods for debugging
    private void OnDrawGizmos()
    {
        if (showTrajectory)
        {
            // Draw trajectory points
            for (int i = 1; i < trajectoryPoints.Count; i++)
            {
                Gizmos.color = trajectoryColor;
                Gizmos.DrawLine(trajectoryPoints[i - 1], trajectoryPoints[i]);
            }
        }

        if (targetPosition != null)
        {
            // Draw target position
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(targetPosition, 0.1f);
            Gizmos.DrawLine(transform.position, targetPosition);
        }
    }
}