#include "GMMFit.h"

Eigen::VectorXd eval(const Eigen::MatrixXd& x, const Eigen::VectorXd mu, const Eigen::MatrixXd sigma) {
  int n = x.cols();  // number of datapoints
  int k = x.rows();  // number of dimensions
  std::cout << sigma.determinant() << std::endl;
  double denom = sqrt(pow(2.0 * M_PI, k) * sigma.determinant());

  std::cout << denom << std::endl;
  Eigen::MatrixXd r = (x).colwise() - mu;
  Eigen::MatrixXd sigma_inv = sigma.inverse();
  Eigen::VectorXd out = Eigen::VectorXd::Zero(n);

  for (int i = 0; i < n; i++) {
    double u = r.col(i).transpose() * sigma_inv * r.col(i);
    out(i) = exp(-0.5 * u);
  }
  out /= denom;
  return out;
}

bool GMMFit::fit_multivariate(
    const Eigen::MatrixXd& in_vec,
    const int num_gmm,
    std::vector<Eigen::VectorXd>& mu,
    std::vector<Eigen::MatrixXd>& sigma,
    Eigen::VectorXd& weights,
    Eigen::MatrixXd& prob) {
  int num_points = in_vec.cols();  // number of data points

  // Run the EM algorithm for a fixed number of iterations
  std::cout << " iter begin " << std::endl;
  for (int itr = 0; itr < 10; itr++) {
    // E-step: Compute the likelihood of each data point under each Gaussian component
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(num_points, num_gmm);

    for (int gs_index = 0; gs_index < num_gmm; gs_index++) {
      Eigen::VectorXd _tmp = eval(in_vec, mu[gs_index], sigma[gs_index]);
      L.col(gs_index) = _tmp;
    }

    for (int i = 0; i < num_points; i++) {
      double p_x = L.row(i).dot(weights);  // total probability (denominator in Bayes' rule)
      prob.row(i) = (L.row(i).array() * weights.array()) / p_x;
    }

    // M-step: Update the parameters for each Gaussian component
    for (int gs_index = 0; gs_index < num_gmm; gs_index++) {
      double denom = prob.col(gs_index).sum();
      Eigen::VectorXd mu_new_numerator = (in_vec * prob.col(gs_index));
      Eigen::VectorXd mu_new = mu_new_numerator / denom;
      Eigen::MatrixXd _tmp_w = (in_vec.colwise() - mu_new).array().rowwise() * prob.col(gs_index).transpose().array();
      Eigen::MatrixXd sigma_new = _tmp_w * (in_vec.colwise() - mu_new).transpose() / denom;

      mu[gs_index] = mu_new;
      sigma[gs_index] = sigma_new;
    }

    // Update weights
    weights = prob.colwise().sum() / num_points;
  }
  return true;
};
