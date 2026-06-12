/*
 * Monte Carlo Tournament Simulation Engine
 * ==========================================
 * High-performance C++ core for simulating the 2026 FIFA World Cup
 * millions of times. Uses Poisson goal model and OpenMP parallelization.
 *
 * Architecture:
 * - MatchProbs: stores W/D/L probabilities + expected goal differential
 * - simulate_match(): samples a scoreline using Poisson distributions
 * - simulate_group(): plays all matches in a group, returns standings
 * - simulate_tournament(): full tournament from groups to final
 * - run_simulation(): parallel Monte Carlo over N iterations
 *
 * Build: compiled via pybind11 (see setup.py / mc_bindings.cpp)
 */

#include <vector>
#include <array>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <cassert>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

// ─── Data Structures ────────────────────────────────────────────────────────

struct MatchProbs {
    double win;     // P(team_a wins)
    double draw;    // P(draw)
    double loss;    // P(team_a loses)
    double xgd;     // Expected goal differential (team_a - team_b)
};

struct MatchResult {
    int goals_a;
    int goals_b;
};

struct TeamStanding {
    int team_id;
    int points;
    int goal_diff;
    int goals_for;
    int goals_against;
    int wins;
    int draws;
    int losses;
};

// Results accumulated across all simulations
struct SimulationResults {
    // Per-team counts: how many times the team reached each round
    std::unordered_map<int, long long> group_exit;
    std::unordered_map<int, long long> round_of_32;
    std::unordered_map<int, long long> round_of_16;
    std::unordered_map<int, long long> quarter_final;
    std::unordered_map<int, long long> semi_final;
    std::unordered_map<int, long long> final_reached;
    std::unordered_map<int, long long> winner;
};

// ─── Utility Functions ──────────────────────────────────────────────────────

static int poisson_sample(double lambda, std::mt19937& rng) {
    /*
     * Sample from Poisson distribution using inverse transform.
     * For lambda < 30, this is efficient. Football goals rarely exceed 8.
     */
    if (lambda <= 0.0) return 0;
    if (lambda > 15.0) lambda = 15.0;  // Cap for safety

    double L = std::exp(-lambda);
    int k = 0;
    double p = 1.0;
    
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    do {
        k++;
        p *= unif(rng);
    } while (p > L);

    return k - 1;
}

// ─── Match Simulation ───────────────────────────────────────────────────────

static MatchResult simulate_match(const MatchProbs& probs, std::mt19937& rng) {
    /*
     * Simulate a single match using the Poisson goal model.
     *
     * Strategy:
     * 1. Convert W/D/L probabilities + xGD into expected goals (lambda)
     *    for each team using a constrained optimization approach.
     * 2. Sample goals from Poisson(lambda_a) and Poisson(lambda_b).
     *
     * The key insight: given xGD = E[goals_a] - E[goals_b] and the 
     * average total goals (~2.5 in international football), we can
     * derive individual lambdas.
     */
    
    // Average total goals per match in international football
    const double avg_total = 2.5;
    
    // Derive individual team expected goals from xGD
    // lambda_a + lambda_b ≈ avg_total
    // lambda_a - lambda_b ≈ xgd
    double lambda_a = (avg_total + probs.xgd) / 2.0;
    double lambda_b = (avg_total - probs.xgd) / 2.0;
    
    // Clamp to reasonable range
    lambda_a = std::max(0.2, std::min(5.0, lambda_a));
    lambda_b = std::max(0.2, std::min(5.0, lambda_b));
    
    // Sample goals
    MatchResult result;
    result.goals_a = poisson_sample(lambda_a, rng);
    result.goals_b = poisson_sample(lambda_b, rng);
    
    return result;
}

static MatchResult simulate_knockout_match(const MatchProbs& probs, std::mt19937& rng) {
    /*
     * Simulate a knockout match. If drawn after 90 minutes,
     * goes to extra time (30 min ~ 1/3 of regular time goals),
     * then penalties (50/50 coin flip with slight advantage to higher-rated).
     */
    MatchResult result = simulate_match(probs, rng);
    
    if (result.goals_a == result.goals_b) {
        // Extra time: ~1/3 chance of a goal per team
        double et_lambda_a = std::max(0.1, (2.5 + probs.xgd) / 6.0);
        double et_lambda_b = std::max(0.1, (2.5 - probs.xgd) / 6.0);
        
        int et_a = poisson_sample(et_lambda_a, rng);
        int et_b = poisson_sample(et_lambda_b, rng);
        result.goals_a += et_a;
        result.goals_b += et_b;
    }
    
    if (result.goals_a == result.goals_b) {
        // Penalties: slightly favor the better team
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        double pk_advantage = 0.5 + (probs.win - probs.loss) * 0.1;
        pk_advantage = std::max(0.35, std::min(0.65, pk_advantage));
        
        if (unif(rng) < pk_advantage) {
            result.goals_a += 1;  // Team A wins on penalties
        } else {
            result.goals_b += 1;  // Team B wins on penalties
        }
    }
    
    return result;
}

// ─── Group Stage Simulation ─────────────────────────────────────────────────

static bool compare_standings(const TeamStanding& a, const TeamStanding& b) {
    /* FIFA tiebreaker rules for group stage */
    if (a.points != b.points) return a.points > b.points;
    if (a.goal_diff != b.goal_diff) return a.goal_diff > b.goal_diff;
    if (a.goals_for != b.goals_for) return a.goals_for > b.goals_for;
    return false;  // If still tied, random order (handled by shuffle)
}

struct GroupResult {
    std::vector<TeamStanding> standings;  // Sorted by position (0 = winner)
};

static GroupResult simulate_group(
    const std::vector<int>& team_ids,
    const std::unordered_map<long long, MatchProbs>& prob_matrix,
    std::mt19937& rng
) {
    /*
     * Simulate all matches in a 4-team group.
     * Each team plays 3 matches (round-robin).
     * Returns sorted standings.
     */
    int n = static_cast<int>(team_ids.size());
    
    // Initialize standings
    std::vector<TeamStanding> standings(n);
    for (int i = 0; i < n; i++) {
        standings[i] = {team_ids[i], 0, 0, 0, 0, 0, 0, 0};
    }

    // Play all matches (round-robin)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            // Look up match probabilities
            long long key = (long long)team_ids[i] * 10000 + team_ids[j];
            auto it = prob_matrix.find(key);
            
            MatchProbs probs;
            if (it != prob_matrix.end()) {
                probs = it->second;
            } else {
                // Default: even match
                probs = {0.35, 0.30, 0.35, 0.0};
            }

            MatchResult result = simulate_match(probs, rng);

            // Update standings for team i
            standings[i].goals_for += result.goals_a;
            standings[i].goals_against += result.goals_b;
            standings[i].goal_diff += (result.goals_a - result.goals_b);
            
            // Update standings for team j
            standings[j].goals_for += result.goals_b;
            standings[j].goals_against += result.goals_a;
            standings[j].goal_diff += (result.goals_b - result.goals_a);

            if (result.goals_a > result.goals_b) {
                standings[i].points += 3;
                standings[i].wins += 1;
                standings[j].losses += 1;
            } else if (result.goals_a == result.goals_b) {
                standings[i].points += 1;
                standings[j].points += 1;
                standings[i].draws += 1;
                standings[j].draws += 1;
            } else {
                standings[j].points += 3;
                standings[j].wins += 1;
                standings[i].losses += 1;
            }
        }
    }

    // Sort standings by FIFA tiebreaker rules
    std::sort(standings.begin(), standings.end(), compare_standings);

    GroupResult gr;
    gr.standings = standings;
    return gr;
}

// ─── Full Tournament Simulation ─────────────────────────────────────────────

struct TournamentConfig {
    int num_groups;                        // 12
    int teams_per_group;                   // 4
    int third_place_advance;              // 8
    std::vector<std::vector<int>> groups;  // group[g] = [team_ids...]
};

static int simulate_full_tournament(
    const TournamentConfig& config,
    const std::unordered_map<long long, MatchProbs>& prob_matrix,
    std::mt19937& rng,
    SimulationResults& results
) {
    /*
     * Simulate the complete 2026 FIFA World Cup:
     * 1. Group Stage: 12 groups of 4 → top 2 + 8 best 3rd advance
     * 2. Round of 32
     * 3. Round of 16
     * 4. Quarter-finals
     * 5. Semi-finals
     * 6. Final
     *
     * Returns the winning team_id.
     */

    // ── Phase 1: Group Stage ────────────────────────────────────────────
    std::vector<GroupResult> group_results(config.num_groups);
    std::vector<int> group_winners;      // 12 group winners
    std::vector<int> group_runners;      // 12 runners-up
    std::vector<TeamStanding> third_placed;  // 12 third-placed teams

    for (int g = 0; g < config.num_groups; g++) {
        group_results[g] = simulate_group(config.groups[g], prob_matrix, rng);
        
        auto& st = group_results[g].standings;
        
        if (st.size() >= 1) group_winners.push_back(st[0].team_id);
        if (st.size() >= 2) group_runners.push_back(st[1].team_id);
        if (st.size() >= 3) {
            st[2].team_id; // group info preserved in team_id
            third_placed.push_back(st[2]);
        }
        
        // Teams that finish 4th exit at group stage
        if (st.size() >= 4) {
            results.group_exit[st[3].team_id]++;
        }
    }

    // Select 8 best third-placed teams
    std::sort(third_placed.begin(), third_placed.end(), compare_standings);
    
    std::vector<int> advancing_thirds;
    for (int i = 0; i < std::min((int)third_placed.size(), config.third_place_advance); i++) {
        advancing_thirds.push_back(third_placed[i].team_id);
    }
    
    // Remaining third-placed teams exit
    for (int i = config.third_place_advance; i < (int)third_placed.size(); i++) {
        results.group_exit[third_placed[i].team_id]++;
    }

    // ── Phase 2: Build Round of 32 bracket ──────────────────────────────
    // 32 teams: 12 winners + 12 runners-up + 8 best thirds
    std::vector<std::pair<int, int>> r32_matchups;
    
    // Simplified bracket: pair winners with thirds/runners from other groups
    // In reality FIFA has a complex mapping, but this captures the key dynamics
    for (int g = 0; g < std::min(8, (int)group_winners.size()); g++) {
        if (g < (int)advancing_thirds.size()) {
            r32_matchups.push_back({group_winners[g], advancing_thirds[g]});
        }
    }
    for (int g = 0; g < std::min((int)group_winners.size(), (int)group_runners.size()); g++) {
        // Pair remaining winners with runners from offset groups
        int runner_idx = (g + 6) % config.num_groups;  // Offset to avoid same group
        if (runner_idx < (int)group_runners.size() && g >= 8) {
            r32_matchups.push_back({group_winners[g], group_runners[runner_idx]});
        }
    }
    // Pair remaining runners with each other
    for (int g = 0; g < (int)group_runners.size(); g += 2) {
        if (g + 1 < (int)group_runners.size()) {
            r32_matchups.push_back({group_runners[g], group_runners[g + 1]});
        }
    }

    // Ensure we have exactly 16 matchups (32 teams)
    // If we don't have enough due to bracket rules, fill remaining
    std::vector<int> all_qualifiers;
    for (auto w : group_winners) all_qualifiers.push_back(w);
    for (auto r : group_runners) all_qualifiers.push_back(r);
    for (auto t : advancing_thirds) all_qualifiers.push_back(t);
    
    // Record R32 qualification
    for (auto tid : all_qualifiers) {
        results.round_of_32[tid]++;
    }
    
    // Rebuild matchups from the qualified list if needed
    if (r32_matchups.size() != 16 && all_qualifiers.size() >= 32) {
        r32_matchups.clear();
        // Standard seeded bracket: 1 vs 32, 2 vs 31, etc.
        for (int i = 0; i < 16; i++) {
            r32_matchups.push_back({all_qualifiers[i], all_qualifiers[31 - i]});
        }
    } else if (r32_matchups.size() != 16) {
        // Fallback: pair sequentially
        r32_matchups.clear();
        for (size_t i = 0; i + 1 < all_qualifiers.size(); i += 2) {
            r32_matchups.push_back({all_qualifiers[i], all_qualifiers[i + 1]});
        }
    }

    // ── Phase 3: Knockout rounds ────────────────────────────────────────
    auto play_knockout_round = [&](
        const std::vector<std::pair<int, int>>& matchups,
        std::unordered_map<int, long long>& next_round_counter
    ) -> std::vector<int> {
        std::vector<int> winners;
        for (auto& [a, b] : matchups) {
            long long key = (long long)std::min(a, b) * 10000 + std::max(a, b);
            auto it = prob_matrix.find(key);
            
            MatchProbs probs;
            if (it != prob_matrix.end()) {
                // If a < b in the key, probs are from a's perspective
                if (a <= b) {
                    probs = it->second;
                } else {
                    probs = {it->second.loss, it->second.draw, it->second.win, -it->second.xgd};
                }
            } else {
                probs = {0.40, 0.20, 0.40, 0.0};
            }

            MatchResult result = simulate_knockout_match(probs, rng);
            int winner_id = (result.goals_a > result.goals_b) ? a : b;
            winners.push_back(winner_id);
            next_round_counter[winner_id]++;
        }
        return winners;
    };

    // Round of 32 → Round of 16
    auto r16_teams = play_knockout_round(r32_matchups, results.round_of_16);
    
    // Build R16 matchups
    std::vector<std::pair<int, int>> r16_matchups;
    for (size_t i = 0; i + 1 < r16_teams.size(); i += 2) {
        r16_matchups.push_back({r16_teams[i], r16_teams[i + 1]});
    }

    // Round of 16 → Quarter-finals
    auto qf_teams = play_knockout_round(r16_matchups, results.quarter_final);
    
    // Build QF matchups
    std::vector<std::pair<int, int>> qf_matchups;
    for (size_t i = 0; i + 1 < qf_teams.size(); i += 2) {
        qf_matchups.push_back({qf_teams[i], qf_teams[i + 1]});
    }

    // Quarter-finals → Semi-finals
    auto sf_teams = play_knockout_round(qf_matchups, results.semi_final);
    
    // Build SF matchups
    std::vector<std::pair<int, int>> sf_matchups;
    for (size_t i = 0; i + 1 < sf_teams.size(); i += 2) {
        sf_matchups.push_back({sf_teams[i], sf_teams[i + 1]});
    }

    // Semi-finals → Final
    auto finalists = play_knockout_round(sf_matchups, results.final_reached);
    
    if (finalists.size() >= 2) {
        // Final
        std::vector<std::pair<int, int>> final_matchup = {{finalists[0], finalists[1]}};
        auto champion = play_knockout_round(final_matchup, results.winner);
        
        if (!champion.empty()) {
            return champion[0];
        }
    } else if (finalists.size() == 1) {
        results.winner[finalists[0]]++;
        return finalists[0];
    }

    return -1;  // Should not happen
}

// ─── Main Simulation Runner ─────────────────────────────────────────────────

SimulationResults run_simulation(
    const TournamentConfig& config,
    const std::unordered_map<long long, MatchProbs>& prob_matrix,
    int num_iterations,
    int num_threads,
    unsigned int base_seed
) {
    /*
     * Run the Monte Carlo simulation with OpenMP parallelization.
     *
     * Each thread gets its own RNG seeded with base_seed + thread_id,
     * ensuring reproducibility while maintaining thread safety.
     */
    
    SimulationResults global_results;

    // Initialize all teams in the results
    for (const auto& group : config.groups) {
        for (int tid : group) {
            global_results.group_exit[tid] = 0;
            global_results.round_of_32[tid] = 0;
            global_results.round_of_16[tid] = 0;
            global_results.quarter_final[tid] = 0;
            global_results.semi_final[tid] = 0;
            global_results.final_reached[tid] = 0;
            global_results.winner[tid] = 0;
        }
    }

    int actual_threads = num_threads;
    #ifdef _OPENMP
    omp_set_num_threads(actual_threads);
    #else
    actual_threads = 1;
    #endif

    // Thread-local results to avoid contention
    std::vector<SimulationResults> thread_results(actual_threads);
    for (auto& tr : thread_results) {
        for (const auto& group : config.groups) {
            for (int tid : group) {
                tr.group_exit[tid] = 0;
                tr.round_of_32[tid] = 0;
                tr.round_of_16[tid] = 0;
                tr.quarter_final[tid] = 0;
                tr.semi_final[tid] = 0;
                tr.final_reached[tid] = 0;
                tr.winner[tid] = 0;
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 100)
    for (int iter = 0; iter < num_iterations; iter++) {
        int thread_id = 0;
        #ifdef _OPENMP
        thread_id = omp_get_thread_num();
        #endif
        
        std::mt19937 rng(base_seed + iter);
        
        simulate_full_tournament(config, prob_matrix, rng, thread_results[thread_id]);
    }

    // Merge thread-local results
    for (const auto& tr : thread_results) {
        for (const auto& [tid, count] : tr.group_exit)
            global_results.group_exit[tid] += count;
        for (const auto& [tid, count] : tr.round_of_32)
            global_results.round_of_32[tid] += count;
        for (const auto& [tid, count] : tr.round_of_16)
            global_results.round_of_16[tid] += count;
        for (const auto& [tid, count] : tr.quarter_final)
            global_results.quarter_final[tid] += count;
        for (const auto& [tid, count] : tr.semi_final)
            global_results.semi_final[tid] += count;
        for (const auto& [tid, count] : tr.final_reached)
            global_results.final_reached[tid] += count;
        for (const auto& [tid, count] : tr.winner)
            global_results.winner[tid] += count;
    }

    return global_results;
}
