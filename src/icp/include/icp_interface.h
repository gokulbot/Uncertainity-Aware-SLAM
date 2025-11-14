

class ICPInterface {
public:
    virtual ~ICPInterface() {}

    // Initialize the ICP algorithm with necessary parameters
    virtual void initialize(const std::string& config_file) = 0;

    // Set the source point cloud
    virtual void setSourcePointCloud(const PointCloud& source) = 0;

    // Set the target point cloud
    virtual void setTargetPointCloud(const PointCloud& target) = 0;

    // Execute the ICP algorithm and return the transformation matrix
    virtual Matrix4f executeICP() = 0;

    // Get the fitness score of the alignment
    virtual float getFitnessScore() const = 0;

    // Get the number of iterations performed
    virtual int getNumIterations() const = 0;
};