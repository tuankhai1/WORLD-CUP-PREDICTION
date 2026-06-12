/*
 * pybind11 Bindings for Monte Carlo Simulation Engine
 * =====================================================
 * Exposes the C++ simulation functions to Python.
 *
 * Python API:
 *   import mc_simulation
 *   results = mc_simulation.simulate(groups, prob_matrix, iterations, threads, seed)
 *   match = mc_simulation.simulate_single_match(win_p, draw_p, loss_p, xgd, seed)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <set>

#include "mc_engine.cpp"

namespace py = pybind11;

// ─── Python-facing wrapper for full tournament simulation ────────────────────

py::dict py_simulate_tournament(
    std::vector<std::vector<int>> groups,         // [[team_ids...], ...]
    std::map<std::string, std::vector<double>> prob_map,  // "id1|id2" -> [win, draw, loss, xgd]
    int num_iterations,
    int num_threads,
    unsigned int seed
) {
    /*
     * Main entry point from Python.
     *
     * Args:
     *   groups: List of groups, each group is a list of team IDs (ints)
     *   prob_map: Dict mapping "teamA_id|teamB_id" -> [win_p, draw_p, loss_p, xgd]
     *   num_iterations: Number of tournament simulations
     *   num_threads: Number of OpenMP threads
     *   seed: Base random seed
     *
     * Returns:
     *   Dict mapping team_id -> {
     *     "group_exit": float (probability),
     *     "r32": float,
     *     "r16": float,
     *     "qf": float,
     *     "sf": float,
     *     "final": float,
     *     "winner": float
     *   }
     */
    
    // Build tournament config
    TournamentConfig config;
    config.num_groups = static_cast<int>(groups.size());
    config.teams_per_group = 4;
    config.third_place_advance = 8;
    config.groups = groups;

    // Convert probability map to C++ format
    // Key encoding: team_id_a * 10000 + team_id_b (where a < b)
    std::unordered_map<long long, MatchProbs> cpp_prob_matrix;
    
    for (const auto& [key, values] : prob_map) {
        if (values.size() < 4) continue;
        
        // Parse "id1|id2" key
        size_t sep = key.find('|');
        if (sep == std::string::npos) continue;
        
        int id1 = std::stoi(key.substr(0, sep));
        int id2 = std::stoi(key.substr(sep + 1));
        
        MatchProbs probs;
        probs.win = values[0];
        probs.draw = values[1];
        probs.loss = values[2];
        probs.xgd = values[3];
        
        // Store with canonical key (smaller id first)
        if (id1 <= id2) {
            long long cpp_key = (long long)id1 * 10000 + id2;
            cpp_prob_matrix[cpp_key] = probs;
        } else {
            long long cpp_key = (long long)id2 * 10000 + id1;
            // Swap perspective
            cpp_prob_matrix[cpp_key] = {probs.loss, probs.draw, probs.win, -probs.xgd};
        }
    }

    // Run simulation
    SimulationResults results = run_simulation(
        config, cpp_prob_matrix, num_iterations, num_threads, seed
    );

    // Convert to Python dict
    double n = static_cast<double>(num_iterations);
    py::dict output;
    
    // Collect all team IDs
    std::set<int> all_teams;
    for (const auto& group : groups) {
        for (int tid : group) all_teams.insert(tid);
    }
    
    for (int tid : all_teams) {
        py::dict team_result;
        team_result["group_exit"] = results.group_exit[tid] / n;
        team_result["r32"] = results.round_of_32[tid] / n;
        team_result["r16"] = results.round_of_16[tid] / n;
        team_result["qf"] = results.quarter_final[tid] / n;
        team_result["sf"] = results.semi_final[tid] / n;
        team_result["final"] = results.final_reached[tid] / n;
        team_result["winner"] = results.winner[tid] / n;
        
        output[py::int_(tid)] = team_result;
    }
    
    return output;
}

// ─── Python-facing wrapper for single match simulation ───────────────────────

py::dict py_simulate_single_match(
    double win_p, double draw_p, double loss_p, double xgd,
    unsigned int seed
) {
    /*
     * Simulate a single match (for debugging / visualization).
     *
     * Returns: {"goals_a": int, "goals_b": int}
     */
    std::mt19937 rng(seed);
    MatchProbs probs = {win_p, draw_p, loss_p, xgd};
    MatchResult result = simulate_match(probs, rng);
    
    py::dict output;
    output["goals_a"] = result.goals_a;
    output["goals_b"] = result.goals_b;
    return output;
}

// ─── Batch single-match simulation for scoreline distribution ────────────────

py::dict py_simulate_match_distribution(
    double win_p, double draw_p, double loss_p, double xgd,
    int num_iterations, unsigned int seed
) {
    /*
     * Simulate a match many times and return the scoreline distribution.
     *
     * Returns: {"scorelines": {"1-0": 0.15, "2-1": 0.12, ...}, 
     *           "avg_goals_a": 1.5, "avg_goals_b": 0.8}
     */
    std::mt19937 rng(seed);
    MatchProbs probs = {win_p, draw_p, loss_p, xgd};
    
    std::map<std::string, int> scoreline_counts;
    long long total_goals_a = 0, total_goals_b = 0;
    
    for (int i = 0; i < num_iterations; i++) {
        MatchResult result = simulate_match(probs, rng);
        total_goals_a += result.goals_a;
        total_goals_b += result.goals_b;
        
        std::string scoreline = std::to_string(result.goals_a) + "-" + 
                                std::to_string(result.goals_b);
        scoreline_counts[scoreline]++;
    }
    
    double n = static_cast<double>(num_iterations);
    
    py::dict scorelines;
    for (const auto& [score, count] : scoreline_counts) {
        scorelines[py::str(score)] = count / n;
    }
    
    py::dict output;
    output["scorelines"] = scorelines;
    output["avg_goals_a"] = total_goals_a / n;
    output["avg_goals_b"] = total_goals_b / n;
    
    return output;
}

// ─── Module Definition ──────────────────────────────────────────────────────

PYBIND11_MODULE(mc_simulation, m) {
    m.doc() = "Monte Carlo tournament simulation engine for World Cup prediction";
    
    m.def("simulate", &py_simulate_tournament,
          "Run full tournament Monte Carlo simulation",
          py::arg("groups"),
          py::arg("prob_map"),
          py::arg("num_iterations") = 1000000,
          py::arg("num_threads") = 4,
          py::arg("seed") = 42);
    
    m.def("simulate_single_match", &py_simulate_single_match,
          "Simulate a single match",
          py::arg("win_p"),
          py::arg("draw_p"),
          py::arg("loss_p"),
          py::arg("xgd"),
          py::arg("seed") = 42);
    
    m.def("simulate_match_distribution", &py_simulate_match_distribution,
          "Simulate a match many times for scoreline distribution",
          py::arg("win_p"),
          py::arg("draw_p"),
          py::arg("loss_p"),
          py::arg("xgd"),
          py::arg("num_iterations") = 100000,
          py::arg("seed") = 42);
}
