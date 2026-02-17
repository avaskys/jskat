package org.jskat.ai.ml;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import org.jskat.control.JSkatEventBus;
import org.jskat.control.SkatGame;
import org.jskat.control.event.table.TableCreatedEvent;
import org.jskat.control.gui.JSkatView;
import org.jskat.control.gui.human.AbstractHumanJSkatPlayer;
import org.jskat.control.iss.ChatMessageType;
import org.jskat.data.*;
import org.jskat.data.iss.ChatMessage;
import org.jskat.data.iss.MoveInformation;
import org.jskat.player.JSkatPlayer;
import org.jskat.util.*;
import org.slf4j.LoggerFactory;

import ai.onnxruntime.OrtException;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Standalone comparison harness that replays identical card distributions
 * through two scenarios to measure the impact of MLPlayerPro vs MLPlayer.
 * <p>
 * Scenario A (Baseline): 3x MLPlayer
 * Scenario B (Pro):       2x MLPlayer + 1x MLPlayerPro (player 3)
 * <p>
 * Games are parallelized: multiple game pairs run concurrently in batches,
 * with A and B games within each pair also running concurrently.
 */
public class MLPlayerComparison {

    /**
     * Usage: MLPlayerComparison [rounds] [key=value ...]
     * <p>
     * Config keys:
     *   a.type           - Player type for scenario A (ml or pro, default: ml)
     *   a.bid-threshold  - Bidding confidence threshold for scenario A (default: 0.70)
     *   b.type           - Player type for P3 in scenario B (ml or pro, default: ml)
     *   b.card-play      - Card play transformer path for P3-B
     *   b.card-play-def  - Defender card play transformer path (optional, overrides card-play when defending)
     *   b.game-eval      - Game eval model path for P3-B (pro only)
     *   b.bidding        - Bidding model path for P3-B ("dense" to skip bidding transformer for pro)
     *   b.bid-threshold  - Bidding confidence threshold for P3-B (default: 0.70)
     *   results          - CSV file for iterative result accumulation
     *   details          - Show per-game detail table (true/false, default: false)
     * <p>
     * Examples:
     *   MLPlayerComparison 36 b.type=pro
     *   MLPlayerComparison 36 b.card-play=/path/to/pro.onnx
     *   MLPlayerComparison 36 b.type=pro b.bid-threshold=0.55
     *   MLPlayerComparison 36 a.type=pro b.type=pro b.card-play=/path/to/v2.onnx
     */
    public static void main(String[] args) {
        // Suppress game engine logging — only show our comparison output
        ((Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME)).setLevel(Level.OFF);

        int rounds = 12;
        Map<String, String> config = new LinkedHashMap<>();

        for (String arg : args) {
            if (arg.contains("=")) {
                String[] parts = arg.split("=", 2);
                config.put(parts[0], parts[1]);
            } else {
                rounds = Integer.parseInt(arg);
            }
        }

        ScenarioConfig configA = buildScenarioConfig("a", config);
        ScenarioConfig configB = buildScenarioConfig("b", config);

        int totalGames = rounds * 3;
        int parallelism = Integer.parseInt(config.getOrDefault("workers",
                String.valueOf(Math.max(1, Runtime.getRuntime().availableProcessors() / 2))));
        parallelism = Math.min(parallelism, totalGames);

        String resultsPath = config.get("results");

        System.out.println("=== ML Player Comparison ===");
        System.out.println("Rounds: " + rounds + " (" + totalGames + " games per scenario)");
        System.out.println("Parallelism: " + parallelism + " workers (" + Runtime.getRuntime().availableProcessors() + " cores)");
        System.out.println("Scenario A: " + configA);
        System.out.println("Scenario B: " + configB);
        if (resultsPath != null) System.out.println("Results file: " + resultsPath);
        System.out.println();

        // Initialize JSkat options
        JSkatOptions options = JSkatOptions.instance(new DesktopSavePathResolver());
        options.resetToDefault();

        // Pre-generate card orderings
        List<List<Card>> deckOrders = new ArrayList<>();
        for (int i = 0; i < totalGames; i++) {
            CardDeck deck = new CardDeck();
            deck.shuffle();
            List<Card> order = new ArrayList<>();
            for (Card card : deck) {
                order.add(card);
            }
            deckOrders.add(order);
        }

        // Load ONNX models once per config — shared across all player instances.
        // Only player state (knowledge, cached decisions) is per-instance.
        SharedModels modelsA = new SharedModels(configA);
        SharedModels modelsB = configA.equals(configB) ? modelsA : new SharedModels(configB);

        String typeNameA = "pro".equals(configA.type()) ? "MLPlayerPro" : "MLPlayer";
        String typeNameB = "pro".equals(configB.type()) ? "MLPlayerPro" : "MLPlayer";

        // Create worker slots — each slot has its own player instances (lightweight,
        // sharing underlying models) so different slots can run games concurrently.
        JSkatPlayer[][] workersA = new JSkatPlayer[parallelism][3];
        JSkatPlayer[][] workersB = new JSkatPlayer[parallelism][3];
        for (int w = 0; w < parallelism; w++) {
            PlayerEntry[] entriesA = createAllPlayers(configA, modelsA);
            PlayerEntry[] entriesB = createP3OverridePlayers(configA, modelsA, configB, modelsB);
            for (int p = 0; p < 3; p++) {
                workersA[w][p] = entriesA[p].player();
                workersB[w][p] = entriesB[p].player();
            }
        }

        // Shared no-op view (stateless, thread-safe)
        NoOpView sharedView = new NoOpView();

        // Global accumulators
        PlayerStats[] statsA = {
                new PlayerStats(typeNameA), new PlayerStats(typeNameA), new PlayerStats(typeNameA)
        };
        PlayerStats[] statsB = {
                new PlayerStats(typeNameA), new PlayerStats(typeNameA), new PlayerStats(typeNameB)
        };
        GameRecord[] allRecordsA = new GameRecord[totalGames];
        GameRecord[] allRecordsB = new GameRecord[totalGames];

        // Thread pool: 2x parallelism so A and B games within each pair run concurrently
        ExecutorService executor = Executors.newFixedThreadPool(parallelism * 2);

        // Load previous results for iterative accumulation
        List<DiffRecord> previousRecords = loadPreviousResults(resultsPath);
        List<DiffRecord> currentRecords = new ArrayList<>();
        int[] gamesCompleted = {0};

        if (!previousRecords.isEmpty()) {
            long prevDiffs = previousRecords.stream()
                    .filter(r -> !"SAME".equals(r.bucket()) && !"PASSED".equals(r.bucket()))
                    .count();
            System.out.println("Loaded " + previousRecords.size() + " previous records (" + prevDiffs + " diffs)");
            System.out.println();
        }

        // Register shutdown hook for Ctrl-C resilience
        Thread shutdownHook = new Thread(() -> {
            try {
                List<DiffRecord> snapshot;
                synchronized (currentRecords) {
                    snapshot = new ArrayList<>(currentRecords);
                }
                System.out.println();
                System.out.println("--- Interrupted at game " + gamesCompleted[0] + "/" + totalGames + " ---");
                int[] dummyLineCount = {0};
                printLiveStats(false, dummyLineCount,
                        gamesCompleted[0], totalGames, statsA, statsB,
                        snapshot, previousRecords);
                System.out.flush();
            } catch (Exception e) {
                System.err.println("Shutdown hook error: " + e.getMessage());
                System.err.flush();
            }
        });
        Runtime.getRuntime().addShutdownHook(shutdownHook);

        int[] liveLineCount = {0};

        // Process games in batches
        for (int batchStart = 0; batchStart < totalGames; batchStart += parallelism) {
            int batchEnd = Math.min(batchStart + parallelism, totalGames);
            int batchLen = batchEnd - batchStart;

            // Submit A and B games for each pair in this batch
            @SuppressWarnings("unchecked")
            CompletableFuture<GameRecord>[] futuresA = new CompletableFuture[batchLen];
            @SuppressWarnings("unchecked")
            CompletableFuture<GameRecord>[] futuresB = new CompletableFuture[batchLen];

            for (int j = 0; j < batchLen; j++) {
                int gameIdx = batchStart + j;
                int w = j;
                int[] seatOrder = computeSeatOrder(gameIdx);
                CardDeck deckA = rebuildDeck(deckOrders.get(gameIdx));
                CardDeck deckB = rebuildDeck(deckOrders.get(gameIdx));

                futuresA[j] = CompletableFuture.supplyAsync(() ->
                        playSingleGame(workersA[w], seatOrder, deckA, "A_" + gameIdx, sharedView), executor);
                futuresB[j] = CompletableFuture.supplyAsync(() ->
                        playSingleGame(workersB[w], seatOrder, deckB, "B_" + gameIdx, sharedView), executor);
            }

            // Collect results and classify
            List<DiffRecord> batchRecords = new ArrayList<>();
            for (int j = 0; j < batchLen; j++) {
                int gameIdx = batchStart + j;
                int[] seatOrder = computeSeatOrder(gameIdx);

                GameRecord recA = futuresA[j].join();
                GameRecord recB = futuresB[j].join();

                allRecordsA[gameIdx] = recA;
                allRecordsB[gameIdx] = recB;

                accumulateStats(statsA, recA, seatOrder);
                accumulateStats(statsB, recB, seatOrder);

                batchRecords.add(classifyGamePair(recA, recB));
            }

            synchronized (currentRecords) {
                currentRecords.addAll(batchRecords);
            }
            appendResults(resultsPath, batchRecords);
            gamesCompleted[0] = batchEnd;

            // Update live display (overwrites in place)
            printLiveStats(liveLineCount[0] > 0, liveLineCount,
                    batchEnd, totalGames, statsA, statsB,
                    currentRecords, previousRecords);
        }

        executor.shutdown();

        // Remove shutdown hook (clean exit)
        try {
            Runtime.getRuntime().removeShutdownHook(shutdownHook);
        } catch (IllegalStateException ignored) {
            // JVM already shutting down
        }

        System.out.println();
        System.out.println();

        // Print final results
        printResults(rounds, totalGames, configA, configB, statsA, statsB);
        boolean showDetails = "true".equals(config.get("details"));
        List<DiffRecord> combinedRecords = new ArrayList<>(previousRecords);
        combinedRecords.addAll(currentRecords);
        printDecisionComparison(Arrays.asList(allRecordsA), Arrays.asList(allRecordsB),
                combinedRecords, showDetails);

        // Close shared models (players don't own them)
        modelsA.close();
        if (modelsB != modelsA) {
            modelsB.close();
        }
    }

    // --- Game Execution ---

    /**
     * Computes the seat order for a given game index.
     * Rotates left each game: game 0 → [0,1,2], game 1 → [1,2,0], game 2 → [2,0,1], ...
     */
    private static int[] computeSeatOrder(int gameIndex) {
        int shift = gameIndex % 3;
        return new int[]{shift, (shift + 1) % 3, (shift + 2) % 3};
    }

    /**
     * Plays a single game and returns the result as a GameRecord.
     * Pure function — no side effects on shared state.
     */
    private static GameRecord playSingleGame(JSkatPlayer[] players, int[] seatOrder,
                                             CardDeck deck, String tableName, JSkatView view) {
        JSkatPlayer foreHand = players[seatOrder[0]];
        JSkatPlayer middleHand = players[seatOrder[1]];
        JSkatPlayer rearHand = players[seatOrder[2]];

        JSkatEventBus.TABLE_EVENT_BUSSES.put(tableName, new EventBus());

        SkatGame game = new SkatGame(tableName, GameVariant.STANDARD,
                foreHand, middleHand, rearHand);
        game.setView(view);
        game.setCardDeck(deck);

        try {
            CompletableFuture.runAsync(game::run).get();
        } catch (Exception e) {
            System.err.println("Game failed: " + e.getMessage());
            return new GameRecord();
        } finally {
            JSkatEventBus.TABLE_EVENT_BUSSES.remove(tableName);
        }

        GameSummary summary = game.getGameSummary();
        SkatGameResult result = game.getGameResult();

        if (summary == null || result == null) {
            System.err.println("Null result");
            return new GameRecord();
        }

        GameType gameType = summary.getGameType();
        Player declarer = summary.getDeclarer();

        GameRecord record = new GameRecord();
        record.gameType = gameType;

        if (gameType != GameType.PASSED_IN) {
            int declarerSeat = positionToIndex(declarer);
            record.declarer = declarer;
            record.declarerPlayerIndex = seatOrder[declarerSeat];
            record.hand = summary.isHand();
            record.gameValue = result.getGameValue();
            record.won = result.isWon();
            record.declarerPoints = result.getFinalDeclarerPoints();
            record.schneider = result.isSchneider();
            record.schwarz = result.isSchwarz();
            record.maxBid = extractMaxBid(game, declarer);

            GameAnnouncement announcement = game.getGameAnnouncement();
            if (announcement != null && announcement.discardedCards() != null) {
                record.discardedCards = announcement.discardedCards();
            }
        }

        return record;
    }

    /**
     * Accumulates a game record into player stats. Called from the main thread only.
     */
    private static void accumulateStats(PlayerStats[] stats, GameRecord record, int[] seatOrder) {
        if (record.gameType == GameType.PASSED_IN) {
            for (int i = 0; i < 3; i++) {
                stats[seatOrder[i]].passedInGames++;
            }
            return;
        }

        int declarerPlayer = record.declarerPlayerIndex;
        stats[declarerPlayer].gamesAsDeclarer++;
        stats[declarerPlayer].totalScore += record.gameValue;
        if (record.won) {
            stats[declarerPlayer].winsAsDeclarer++;
        }
        stats[declarerPlayer].gameTypeCounts.merge(record.gameType, 1, Integer::sum);
        if (record.schneider) stats[declarerPlayer].schneiderCount++;
        if (record.schwarz) stats[declarerPlayer].schwarzCount++;

        for (int pos = 0; pos < 3; pos++) {
            int playerIdx = seatOrder[pos];
            if (playerIdx != declarerPlayer) {
                stats[playerIdx].gamesAsOpponent++;
                if (!record.won) {
                    stats[playerIdx].winsAsOpponent++;
                }
            }
        }
    }

    private static void closePlayer(JSkatPlayer player) {
        if (player instanceof AbstractMLPlayer mlPlayer) {
            mlPlayer.close();
        }
    }

    // --- Output Formatting ---

    private static String formatLastGame(GameRecord rec) {
        if (rec.gameType == GameType.PASSED_IN) return "--";
        String prefix = rec.won ? "+" : "-";
        return prefix + Math.abs(rec.gameValue);
    }

    // --- Deck Helpers ---

    private static CardDeck rebuildDeck(List<Card> order) {
        List<Card> fh = new ArrayList<>();
        List<Card> mh = new ArrayList<>();
        List<Card> rh = new ArrayList<>();
        List<Card> skat = new ArrayList<>();

        fh.addAll(order.subList(0, 3));
        mh.addAll(order.subList(3, 6));
        rh.addAll(order.subList(6, 9));
        skat.addAll(order.subList(9, 11));
        fh.addAll(order.subList(11, 15));
        mh.addAll(order.subList(15, 19));
        rh.addAll(order.subList(19, 23));
        fh.addAll(order.subList(23, 26));
        mh.addAll(order.subList(26, 29));
        rh.addAll(order.subList(29, 32));

        return new CardDeck(fh, mh, rh, skat);
    }

    private static int extractMaxBid(SkatGame game, Player declarer) {
        // Note: bid events are posted as TableGameMoveEvent (not SkatGameEvent),
        // so they don't appear in game.getGameMoves(). Use the direct accessor.
        return game.getMaxPlayerBid(declarer);
    }

    private static int positionToIndex(Player position) {
        return switch (position) {
            case FOREHAND -> 0;
            case MIDDLEHAND -> 1;
            case REARHAND -> 2;
        };
    }

    // --- Results ---

    private static void printResults(int rounds, int totalGames,
                                     ScenarioConfig configA, ScenarioConfig configB,
                                     PlayerStats[] statsA, PlayerStats[] statsB) {
        System.out.println("=== Final Results ===");
        System.out.println("Rounds: " + rounds + " (" + totalGames + " games per scenario)");
        System.out.println();

        System.out.println("--- Scenario A: " + configA + " ---");
        for (int i = 0; i < 3; i++) {
            printPlayerLine("Player " + (i + 1), statsA[i]);
        }

        System.out.println();
        System.out.println("--- Scenario B: " + configB + " ---");
        for (int i = 0; i < 3; i++) {
            printPlayerLine("Player " + (i + 1), statsB[i]);
        }

        System.out.println();
        System.out.println("--- Comparison (Player 3: same cards, different config) ---");

        PlayerStats p3A = statsA[2];
        PlayerStats p3B = statsB[2];

        double p3AWinRate = p3A.gamesAsDeclarer > 0
                ? 100.0 * p3A.winsAsDeclarer / p3A.gamesAsDeclarer : 0;
        double p3BWinRate = p3B.gamesAsDeclarer > 0
                ? 100.0 * p3B.winsAsDeclarer / p3B.gamesAsDeclarer : 0;
        double p3AAvgValue = p3A.gamesAsDeclarer > 0
                ? (double) p3A.totalScore / p3A.gamesAsDeclarer : 0;
        double p3BAvgValue = p3B.gamesAsDeclarer > 0
                ? (double) p3B.totalScore / p3B.gamesAsDeclarer : 0;

        System.out.printf("Score:        A=%d  B=%d  (diff %+d)%n",
                p3A.totalScore, p3B.totalScore, p3B.totalScore - p3A.totalScore);
        System.out.printf("Win rate:     A=%.1f%%  B=%.1f%%  (diff %+.1f%%)%n",
                p3AWinRate, p3BWinRate, p3BWinRate - p3AWinRate);
        System.out.printf("Avg value:    A=%.1f  B=%.1f  (diff %+.1f)%n",
                p3AAvgValue, p3BAvgValue, p3BAvgValue - p3AAvgValue);
        System.out.printf("As opponent:  A wins=%d  B wins=%d%n",
                p3A.winsAsOpponent, p3B.winsAsOpponent);

        System.out.println();
        System.out.print("Game types (P3-A): ");
        p3A.gameTypeCounts.forEach((type, count) -> System.out.print(type + "=" + count + " "));
        System.out.println();
        System.out.print("Game types (P3-B): ");
        p3B.gameTypeCounts.forEach((type, count) -> System.out.print(type + "=" + count + " "));
        System.out.println();

        int passedA = statsA[0].passedInGames;
        int passedB = statsB[0].passedInGames;
        System.out.println("Passed-in games: A=" + passedA + ", B=" + passedB);
    }

    private static void printPlayerLine(String label, PlayerStats stats) {
        double winRate = stats.gamesAsDeclarer > 0
                ? 100.0 * stats.winsAsDeclarer / stats.gamesAsDeclarer : 0;
        double avgValue = stats.gamesAsDeclarer > 0
                ? (double) stats.totalScore / stats.gamesAsDeclarer : 0;

        System.out.printf("  %s (%s): Score=%-6d Declared=%-3d Won=%-3d (%.0f%%) AvgValue=%.1f " +
                        "Opponent=%-3d OppWins=%-3d Schneider=%-2d Schwarz=%-2d%n",
                label, stats.playerType,
                stats.totalScore,
                stats.gamesAsDeclarer, stats.winsAsDeclarer, winRate, avgValue,
                stats.gamesAsOpponent, stats.winsAsOpponent,
                stats.schneiderCount, stats.schwarzCount);
    }

    // --- Decision Comparison ---

    private static final String[] BUCKET_NAMES = {"BID", "GAME", "HAND", "DISC", "PLAY-D", "PLAY-F"};

    private static void printDecisionComparison(List<GameRecord> recordsA, List<GameRecord> recordsB,
                                                 List<DiffRecord> combinedDiffRecords, boolean showDetails) {
        System.out.println();
        System.out.println("--- Per-Game Decision Comparison (Player 3 focus) ---");

        int total = recordsA.size();
        int bothPassedIn = 0;
        int sameOutcome = 0;
        int[] bucketCounts = new int[BUCKET_NAMES.length];
        int[] bucketImpacts = new int[BUCKET_NAMES.length];

        record DetailRow(int gameNum, int bucket, String role, String descA, String descB, int impact) {}
        List<DetailRow> details = new ArrayList<>();

        for (int i = 0; i < total; i++) {
            GameRecord a = recordsA.get(i);
            GameRecord b = recordsB.get(i);

            boolean aPassedIn = a.gameType == GameType.PASSED_IN;
            boolean bPassedIn = b.gameType == GameType.PASSED_IN;

            if (aPassedIn && bPassedIn) {
                bothPassedIn++;
                continue;
            }

            if (aPassedIn != bPassedIn) {
                int impact = proImpactForBidDiff(a, b);
                String aDesc = aPassedIn ? "PASSED_IN"
                        : String.format("P%d %s %s %d", a.declarerPlayerIndex + 1,
                        a.gameType, a.won ? "W" : "L", a.gameValue);
                String bDesc = bPassedIn ? "PASSED_IN"
                        : String.format("P%d %s %s %d", b.declarerPlayerIndex + 1,
                        b.gameType, b.won ? "W" : "L", b.gameValue);
                bucketCounts[0]++;
                bucketImpacts[0] += impact;
                details.add(new DetailRow(i + 1, 0, "--", aDesc, bDesc, impact));
                continue;
            }

            boolean p3DeclA = a.declarerPlayerIndex == 2;
            boolean p3DeclB = b.declarerPlayerIndex == 2;
            String role = p3DeclA || p3DeclB ? "decl" : "def";

            if (a.declarerPlayerIndex != b.declarerPlayerIndex) {
                int impact = proImpactForBidDiff(a, b);
                String aDesc = String.format("P%d %s %s %d", a.declarerPlayerIndex + 1,
                        a.gameType, a.won ? "W" : "L", a.gameValue);
                String bDesc = String.format("P%d %s %s %d", b.declarerPlayerIndex + 1,
                        b.gameType, b.won ? "W" : "L", b.gameValue);
                bucketCounts[0]++;
                bucketImpacts[0] += impact;
                details.add(new DetailRow(i + 1, 0, "--", aDesc, bDesc, impact));
            } else if (a.maxBid != b.maxBid && a.gameValue != b.gameValue) {
                int impact = computeProImpact(a, b, p3DeclA);
                String aDesc = String.format("P%d %s(%d) %s %d", a.declarerPlayerIndex + 1,
                        a.gameType, a.maxBid, a.won ? "W" : "L", a.gameValue);
                String bDesc = String.format("P%d %s(%d) %s %d", b.declarerPlayerIndex + 1,
                        b.gameType, b.maxBid, b.won ? "W" : "L", b.gameValue);
                bucketCounts[0]++;
                bucketImpacts[0] += impact;
                details.add(new DetailRow(i + 1, 0, role, aDesc, bDesc, impact));
            } else if (a.gameType != b.gameType) {
                int impact = computeProImpact(a, b, p3DeclA);
                String aDesc = String.format("%s %s %d", a.gameType, a.won ? "W" : "L", a.gameValue);
                String bDesc = String.format("%s %s %d", b.gameType, b.won ? "W" : "L", b.gameValue);
                bucketCounts[1]++;
                bucketImpacts[1] += impact;
                details.add(new DetailRow(i + 1, 1, role, aDesc, bDesc, impact));
            } else if (a.hand != b.hand) {
                int impact = computeProImpact(a, b, p3DeclA);
                String aDesc = String.format("%s %s %s %d", a.gameType, a.hand ? "hand" : "pickup",
                        a.won ? "W" : "L", a.gameValue);
                String bDesc = String.format("%s %s %s %d", b.gameType, b.hand ? "hand" : "pickup",
                        b.won ? "W" : "L", b.gameValue);
                bucketCounts[2]++;
                bucketImpacts[2] += impact;
                details.add(new DetailRow(i + 1, 2, role, aDesc, bDesc, impact));
            } else if (a.gameValue != b.gameValue && !a.hand && !Objects.equals(a.discardedCards, b.discardedCards)) {
                int impact = computeProImpact(a, b, p3DeclA);
                String aDesc = String.format("%s %s %d %s", a.gameType, a.won ? "W" : "L",
                        a.gameValue, a.discardedCards);
                String bDesc = String.format("%s %s %d %s", b.gameType, b.won ? "W" : "L",
                        b.gameValue, b.discardedCards);
                bucketCounts[3]++;
                bucketImpacts[3] += impact;
                details.add(new DetailRow(i + 1, 3, role, aDesc, bDesc, impact));
            } else if (a.gameValue != b.gameValue) {
                int impact = computeProImpact(a, b, p3DeclA);
                String aDesc = String.format("%s %s %d (%dpts)", a.gameType, a.won ? "W" : "L",
                        a.gameValue, a.declarerPoints);
                String bDesc = String.format("%s %s %d (%dpts)", b.gameType, b.won ? "W" : "L",
                        b.gameValue, b.declarerPoints);
                int bucket = p3DeclA ? 4 : 5; // PLAY-D or PLAY-F
                bucketCounts[bucket]++;
                bucketImpacts[bucket] += impact;
                details.add(new DetailRow(i + 1, bucket, role, aDesc, bDesc, impact));
            } else {
                sameOutcome++;
            }
        }

        int totalDiffs = details.size();
        int totalImpact = 0;
        int proWins = 0;
        int proLosses = 0;
        for (DetailRow d : details) {
            totalImpact += d.impact;
            if (d.impact > 0) proWins++;
            else if (d.impact < 0) proLosses++;
        }

        System.out.printf("This run: %d games  |  Same outcome: %d  |  Both passed in: %d  |  Differences: %d%n",
                total, sameOutcome, bothPassedIn, totalDiffs);

        // Combined bucket totals (previous + current)
        Map<String, List<Integer>> combinedBucketImpacts = new LinkedHashMap<>();
        for (String name : BUCKET_NAMES) combinedBucketImpacts.put(name, new ArrayList<>());
        List<Integer> declImpacts = new ArrayList<>();
        List<Integer> defImpacts = new ArrayList<>();
        for (DiffRecord r : combinedDiffRecords) {
            List<Integer> list = combinedBucketImpacts.get(r.bucket());
            if (list != null) list.add(r.impact());
            if ("SAME".equals(r.bucket()) || "PASSED".equals(r.bucket())) continue;
            if (r.p3Declaring()) declImpacts.add(r.impact());
            else defImpacts.add(r.impact());
        }
        int combinedTotalDiffs = 0;
        int combinedTotalImpact = 0;
        for (List<Integer> impacts : combinedBucketImpacts.values()) {
            combinedTotalDiffs += impacts.size();
            for (int v : impacts) combinedTotalImpact += v;
        }

        boolean hasPrevious = combinedTotalDiffs > totalDiffs;
        System.out.println();
        System.out.printf("  %-6s  %5s  %8s  %9s  %s%n", "Type", "n", "Impact", "Mean/game", "95% CI");
        System.out.println("  " + "-".repeat(56));
        for (int b = 0; b < BUCKET_NAMES.length; b++) {
            List<Integer> impacts = combinedBucketImpacts.get(BUCKET_NAMES[b]);
            if (!impacts.isEmpty()) {
                System.out.println(formatCILine(BUCKET_NAMES[b], impacts));
            }
        }
        System.out.println("  " + "-".repeat(56));
        List<Integer> allImpacts = new ArrayList<>();
        for (List<Integer> impacts : combinedBucketImpacts.values()) allImpacts.addAll(impacts);
        if (!allImpacts.isEmpty()) {
            System.out.println(formatCILine("ALL", allImpacts));
        }
        System.out.printf("  Pro better: %d  |  Pro worse: %d%n", proWins, proLosses);

        if (declImpacts.size() + defImpacts.size() >= 2) {
            System.out.println();
            if (hasPrevious) System.out.println("  (CI uses all accumulated data)");
            printCI("Declaring", declImpacts);
            printCI("Defending", defImpacts);
        }

        if (showDetails && !details.isEmpty()) {
            System.out.println();
            System.out.printf("  %4s  %-6s  %-4s  %-26s  %-26s  %s%n",
                    "#", "Type", "Role", "Baseline (A)", "Pro (B)", "Pro Impact");
            System.out.println("  " + "-".repeat(90));
            for (DetailRow d : details) {
                System.out.printf("  %4d  %-6s  %-4s  %-26s  %-26s  %s%n",
                        d.gameNum, BUCKET_NAMES[d.bucket], d.role, d.descA, d.descB,
                        formatImpact(d.impact));
            }
        }
    }

    private static void printCI(String label, List<Integer> impacts) {
        System.out.println(formatCILine(label, impacts));
    }

    private static int computeProImpact(GameRecord a, GameRecord b, boolean p3Declaring) {
        if (p3Declaring) {
            return b.gameValue - a.gameValue;
        } else {
            return a.gameValue - b.gameValue;
        }
    }

    private static int proImpactForBidDiff(GameRecord a, GameRecord b) {
        boolean p3DeclaredA = a.declarerPlayerIndex == 2;
        boolean p3DeclaredB = b.declarerPlayerIndex == 2;
        if (p3DeclaredB && !p3DeclaredA) {
            return b.gameValue;
        } else if (p3DeclaredA && !p3DeclaredB) {
            return -a.gameValue;
        } else {
            return b.gameValue - a.gameValue;
        }
    }

    private static String formatImpact(int impact) {
        if (impact > 0) return String.format("+%d ^", impact);
        if (impact < 0) return String.format("%d v", impact);
        return "0";
    }

    // --- Configuration ---

    private static String modelPath(String filename) {
        return AbstractMLPlayer.getDefaultModelPath(filename);
    }

    private record ScenarioConfig(String type, String cardPlay, String cardPlayDef,
                                  String gameEval, String bidding, float bidThreshold) {
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder(type);
            List<String> overrides = new ArrayList<>();
            if (cardPlay != null) overrides.add("cp=" + shortPath(cardPlay));
            if (cardPlayDef != null) overrides.add("cpd=" + shortPath(cardPlayDef));
            if (gameEval != null) overrides.add("ge=" + shortPath(gameEval));
            if (bidding != null) overrides.add("bid=" + shortPath(bidding));
            if (bidThreshold != 0.70f) overrides.add("thr=" + bidThreshold);
            if (!overrides.isEmpty()) sb.append(" (").append(String.join(", ", overrides)).append(")");
            return sb.toString();
        }

        private static String shortPath(String path) {
            int sep = path.lastIndexOf('/');
            return sep >= 0 ? path.substring(sep + 1) : path;
        }
    }

    private static ScenarioConfig buildScenarioConfig(String prefix, Map<String, String> config) {
        String type = config.getOrDefault(prefix + ".type", "ml");
        String cardPlay = config.get(prefix + ".card-play");
        String cardPlayDef = config.get(prefix + ".card-play-def");
        String gameEval = config.get(prefix + ".game-eval");
        String bidding = config.get(prefix + ".bidding");
        float bidThreshold = Float.parseFloat(config.getOrDefault(prefix + ".bid-threshold", "0.70"));
        return new ScenarioConfig(type, cardPlay, cardPlayDef, gameEval, bidding, bidThreshold);
    }

    /**
     * Holds shared ONNX model instances for a scenario config.
     * Models are thread-safe (OrtSession supports concurrent run() calls).
     * Created once, shared across all player instances with matching config.
     */
    private static class SharedModels implements AutoCloseable {
        final ONNXModelWrapper biddingDense;
        final TransformerModelWrapper cardPlayTransformer;
        final TransformerModelWrapper cardPlayDefTransformer; // nullable
        // MLPlayer-specific
        final ONNXModelWrapper gameEvalDense;
        // MLPlayerPro-specific
        final CardSetEvaluatorWrapper gameEvalTransformer;
        final PreSkatTransformerWrapper biddingTransformer;

        SharedModels(ScenarioConfig config) {
            try {
                if ("pro".equals(config.type)) {
                    this.biddingDense = new ONNXModelWrapper(
                            modelPath("bidding_dense.onnx"), ONNXModelWrapper.ModelType.BIDDING_DENSE);
                    String ge = config.gameEval != null ? config.gameEval : modelPath("game_eval_transformer.onnx");
                    String cp = config.cardPlay != null ? config.cardPlay : modelPath("card_play_transformer_pro.onnx");
                    String bid = config.bidding != null ? config.bidding : modelPath("bidding_transformer.onnx");
                    this.cardPlayTransformer = new TransformerModelWrapper(cp);
                    this.gameEvalDense = null;
                    this.gameEvalTransformer = new CardSetEvaluatorWrapper(ge);
                    PreSkatTransformerWrapper bt = null;
                    if (!"dense".equals(config.bidding)) {
                        try { bt = new PreSkatTransformerWrapper(bid); } catch (Exception ignored) {}
                    }
                    this.biddingTransformer = bt;
                } else {
                    String bd = config.bidding != null ? config.bidding : modelPath("bidding_dense.onnx");
                    String ge = config.gameEval != null ? config.gameEval : modelPath("game_eval_dense.onnx");
                    String cp = config.cardPlay != null ? config.cardPlay : modelPath("card_play_transformer.onnx");
                    this.biddingDense = new ONNXModelWrapper(bd, ONNXModelWrapper.ModelType.BIDDING_DENSE);
                    this.cardPlayTransformer = new TransformerModelWrapper(cp);
                    this.gameEvalDense = new ONNXModelWrapper(ge, ONNXModelWrapper.ModelType.GAME_EVAL_DENSE);
                    this.gameEvalTransformer = null;
                    this.biddingTransformer = null;
                }
                // Optional defender card play model
                if (config.cardPlayDef != null) {
                    this.cardPlayDefTransformer = new TransformerModelWrapper(config.cardPlayDef);
                } else {
                    this.cardPlayDefTransformer = null;
                }
            } catch (OrtException | IOException e) {
                throw new RuntimeException("Failed to load models", e);
            }
        }

        @Override
        public void close() {
            if (biddingDense != null) biddingDense.close();
            if (cardPlayTransformer != null) cardPlayTransformer.close();
            if (cardPlayDefTransformer != null) cardPlayDefTransformer.close();
            if (gameEvalDense != null) gameEvalDense.close();
            if (gameEvalTransformer != null) gameEvalTransformer.close();
            if (biddingTransformer != null) biddingTransformer.close();
        }
    }

    private static AbstractMLPlayer createPlayer(ScenarioConfig config, SharedModels models) {
        AbstractMLPlayer player;
        if ("pro".equals(config.type)) {
            player = new MLPlayerPro(models.biddingDense, models.gameEvalTransformer,
                    models.cardPlayTransformer, models.biddingTransformer);
        } else {
            player = new MLPlayer(models.biddingDense, models.gameEvalDense,
                    models.cardPlayTransformer);
        }
        player.setBidConfidenceThreshold(config.bidThreshold);
        if (models.cardPlayDefTransformer != null) {
            player.setDefenderCardPlayModel(models.cardPlayDefTransformer);
        }
        return player;
    }

    /** Creates 3 identical players from the given config, sharing models. */
    private static PlayerEntry[] createAllPlayers(ScenarioConfig config, SharedModels models) {
        String typeName = "pro".equals(config.type) ? "MLPlayerPro" : "MLPlayer";
        return new PlayerEntry[]{
                new PlayerEntry(createPlayer(config, models), typeName),
                new PlayerEntry(createPlayer(config, models), typeName),
                new PlayerEntry(createPlayer(config, models), typeName)
        };
    }

    /** Creates P1+P2 from configA, P3 from configB, sharing models. */
    private static PlayerEntry[] createP3OverridePlayers(ScenarioConfig configA, SharedModels modelsA,
                                                         ScenarioConfig configB, SharedModels modelsB) {
        String typeA = "pro".equals(configA.type) ? "MLPlayerPro" : "MLPlayer";
        String typeB = "pro".equals(configB.type) ? "MLPlayerPro" : "MLPlayer";
        return new PlayerEntry[]{
                new PlayerEntry(createPlayer(configA, modelsA), typeA),
                new PlayerEntry(createPlayer(configA, modelsA), typeA),
                new PlayerEntry(createPlayer(configB, modelsB), typeB)
        };
    }

    private record PlayerEntry(JSkatPlayer player, String typeName) {}

    // --- Data Classes ---

    private static class GameRecord {
        Player declarer;
        int declarerPlayerIndex = -1;
        GameType gameType = GameType.PASSED_IN;
        boolean hand;
        int maxBid;
        int gameValue;
        boolean won;
        int declarerPoints;
        CardList discardedCards = CardList.empty();
        boolean schneider;
        boolean schwarz;
    }

    private record DiffRecord(String bucket, int impact, boolean p3Declaring) {}

    /**
     * Classifies a game pair into a DiffRecord. Returns null if both games had the same outcome.
     */
    private static DiffRecord classifyGamePair(GameRecord a, GameRecord b) {
        boolean aPassedIn = a.gameType == GameType.PASSED_IN;
        boolean bPassedIn = b.gameType == GameType.PASSED_IN;

        if (aPassedIn && bPassedIn) {
            return new DiffRecord("PASSED", 0, false);
        }

        if (aPassedIn != bPassedIn) {
            int impact = proImpactForBidDiff(a, b);
            boolean p3Decl = a.declarerPlayerIndex == 2 || b.declarerPlayerIndex == 2;
            return new DiffRecord("BID", impact, p3Decl);
        }

        boolean p3DeclA = a.declarerPlayerIndex == 2;
        boolean p3DeclB = b.declarerPlayerIndex == 2;
        boolean p3Decl = p3DeclA || p3DeclB;

        if (a.declarerPlayerIndex != b.declarerPlayerIndex) {
            return new DiffRecord("BID", proImpactForBidDiff(a, b), p3Decl);
        } else if (a.maxBid != b.maxBid && a.gameValue != b.gameValue) {
            return new DiffRecord("BID", computeProImpact(a, b, p3DeclA), p3Decl);
        } else if (a.gameType != b.gameType) {
            return new DiffRecord("GAME", computeProImpact(a, b, p3DeclA), p3Decl);
        } else if (a.hand != b.hand) {
            return new DiffRecord("HAND", computeProImpact(a, b, p3DeclA), p3Decl);
        } else if (a.gameValue != b.gameValue && !a.hand && !Objects.equals(a.discardedCards, b.discardedCards)) {
            return new DiffRecord("DISC", computeProImpact(a, b, p3DeclA), p3Decl);
        } else if (a.gameValue != b.gameValue) {
            String bucket = p3DeclA ? "PLAY-D" : "PLAY-F";
            return new DiffRecord(bucket, computeProImpact(a, b, p3DeclA), p3Decl);
        } else if (a.gameValue == b.gameValue && a.declarerPlayerIndex == b.declarerPlayerIndex
                && a.gameType == b.gameType && a.hand == b.hand && a.won == b.won) {
            return new DiffRecord("SAME", 0, false);
        }

        return new DiffRecord("SAME", 0, false);
    }

    // --- CSV Persistence ---

    private static List<DiffRecord> loadPreviousResults(String path) {
        List<DiffRecord> records = new ArrayList<>();
        if (path == null) return records;
        Path file = Paths.get(path);
        if (!Files.exists(file)) return records;
        try {
            for (String line : Files.readAllLines(file)) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("bucket")) continue; // skip header
                String[] parts = line.split(",", 3);
                if (parts.length == 3) {
                    records.add(new DiffRecord(parts[0], Integer.parseInt(parts[1]),
                            Boolean.parseBoolean(parts[2])));
                }
            }
        } catch (IOException e) {
            System.err.println("Warning: could not read previous results from " + path + ": " + e.getMessage());
        }
        return records;
    }

    private static void appendResults(String path, List<DiffRecord> records) {
        if (path == null || records.isEmpty()) return;
        Path file = Paths.get(path);
        try {
            boolean needsHeader = !Files.exists(file) || Files.size(file) == 0;
            try (BufferedWriter w = Files.newBufferedWriter(file, StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND)) {
                if (needsHeader) {
                    w.write("bucket,impact,p3decl");
                    w.newLine();
                }
                for (DiffRecord r : records) {
                    w.write(r.bucket() + "," + r.impact() + "," + r.p3Declaring());
                    w.newLine();
                }
            }
        } catch (IOException e) {
            System.err.println("Warning: could not write results to " + path + ": " + e.getMessage());
        }
    }

    // --- Live Stats Display ---

    private static final String[] DIFF_BUCKET_NAMES = {"BID", "GAME", "HAND", "DISC", "PLAY-D", "PLAY-F"};

    private static boolean isDiff(DiffRecord r) {
        return !"SAME".equals(r.bucket()) && !"PASSED".equals(r.bucket());
    }

    private static int countDiffs(List<DiffRecord> records) {
        int count = 0;
        for (DiffRecord r : records) {
            if (isDiff(r)) count++;
        }
        return count;
    }

    /**
     * Prints a live stats block that overwrites itself using ANSI cursor-up.
     * Vertical layout — one concept per line.
     */
    private static void printLiveStats(boolean overwrite, int[] lineCount,
                                        int gamesCompleted, int totalGames,
                                        PlayerStats[] statsA, PlayerStats[] statsB,
                                        List<DiffRecord> currentRecords, List<DiffRecord> previousRecords) {
        int currentDiffs = countDiffs(currentRecords);
        int prevDiffs = countDiffs(previousRecords);

        List<DiffRecord> combined = new ArrayList<>(currentRecords);
        combined.addAll(previousRecords);

        Map<String, List<Integer>> buckets = new LinkedHashMap<>();
        for (String name : DIFF_BUCKET_NAMES) buckets.put(name, new ArrayList<>());
        List<Integer> declImpacts = new ArrayList<>();
        List<Integer> defImpacts = new ArrayList<>();
        for (DiffRecord r : combined) {
            List<Integer> list = buckets.get(r.bucket());
            if (list != null) list.add(r.impact());
            if (!isDiff(r)) continue;
            if (r.p3Declaring()) declImpacts.add(r.impact());
            else defImpacts.add(r.impact());
        }

        // Build output lines
        List<String> lines = new ArrayList<>();

        String prevStr = prevDiffs > 0 ? "  (+" + prevDiffs + " previous)" : "";
        lines.add(String.format("  Game %d/%d (%.0f%%)    %d diffs%s",
                gamesCompleted, totalGames,
                100.0 * gamesCompleted / totalGames,
                currentDiffs, prevStr));
        lines.add("");
        lines.add(String.format("  P3 score     A: %-6d   B: %-6d   (%+d)",
                statsA[2].totalScore, statsB[2].totalScore,
                statsB[2].totalScore - statsA[2].totalScore));
        lines.add("");
        lines.add(String.format("  %-9s  %5s  %8s  %9s  %s", "Type", "n", "Impact", "Mean/game", "95% CI"));
        lines.add("  " + "-".repeat(56));
        for (var entry : buckets.entrySet()) {
            List<Integer> impacts = entry.getValue();
            if (!impacts.isEmpty()) {
                lines.add(formatCILine(entry.getKey(), impacts));
            }
        }
        lines.add("  " + "-".repeat(56));
        if (declImpacts.size() >= 2) {
            lines.add(formatCILine("Declaring", declImpacts));
        }
        if (defImpacts.size() >= 2) {
            lines.add(formatCILine("Defending", defImpacts));
        }

        // Move cursor up to overwrite previous display
        if (overwrite && lineCount[0] > 0) {
            System.out.print("\033[" + lineCount[0] + "A");
        }

        for (String line : lines) {
            System.out.print("\033[2K");
            System.out.println(line);
        }
        // Clear leftover lines from previous (longer) display
        for (int i = lines.size(); i < lineCount[0]; i++) {
            System.out.print("\033[2K\n");
        }
        if (lines.size() < lineCount[0]) {
            System.out.print("\033[" + (lineCount[0] - lines.size()) + "A");
        }

        lineCount[0] = lines.size();
        System.out.flush();
    }

    private static String formatCILine(String label, List<Integer> impacts) {
        int n = impacts.size();
        double sum = 0;
        for (int v : impacts) sum += v;
        if (n < 2) {
            return String.format("  %-9s  %5d  %+8.0f", label, n, sum);
        }
        double mean = sum / n;
        double sumSqDiff = 0;
        for (int v : impacts) {
            double diff = v - mean;
            sumSqDiff += diff * diff;
        }
        double stdErr = Math.sqrt(sumSqDiff / (n - 1)) / Math.sqrt(n);
        double ciLow = mean - 1.96 * stdErr;
        double ciHigh = mean + 1.96 * stdErr;
        return String.format("  %-9s  %5d  %+8.0f  %+9.1f  [%+.1f, %+.1f]",
                label, n, sum, mean, ciLow, ciHigh);
    }

    private static class PlayerStats {
        final String playerType;
        int totalScore = 0;
        int gamesAsDeclarer = 0;
        int winsAsDeclarer = 0;
        int gamesAsOpponent = 0;
        int winsAsOpponent = 0;
        int schneiderCount = 0;
        int schwarzCount = 0;
        int passedInGames = 0;
        final Map<GameType, Integer> gameTypeCounts = new EnumMap<>(GameType.class);

        PlayerStats(String playerType) {
            this.playerType = playerType;
        }
    }

    /**
     * Minimal no-op view for headless game execution. Thread-safe (no mutable state).
     */
    private static class NoOpView implements JSkatView {
        NoOpView() {
            JSkatEventBus.INSTANCE.register(this);
        }

        @Subscribe
        public void handle(final TableCreatedEvent event) {
            // no-op
        }

        @Override public String getNewTableName(int localTablesCreated) { return "Table"; }
        @Override public void startGame(String tableName) {}
        @Override public List<String> getPlayerForInvitation(Set<String> playerNames) { return null; }
        @Override public void showMessage(String title, String message) {}
        @Override public void showErrorMessage(String title, String message) {}
        @Override public void showCardNotAllowedMessage(Card card) {}
        @Override public void appendISSChatMessage(ChatMessageType messageType, ChatMessage message) {}
        @Override public void updateISSMove(String tableName, SkatGameData gameData, MoveInformation moveInformation) {}
        @Override public void setResign(String tableName, Player player) {}
        @Override public boolean showISSTableInvitation(String invitor, String tableName) { return false; }
        @Override public void setGeschoben(String tableName, Player player) {}
        @Override public void setDiscardedSkat(String tableName, Player activePlayer, CardList skatBefore, CardList discardedSkat) {}
        @Override public void openWebPage(String link) {}
        @Override public AbstractHumanJSkatPlayer getHumanPlayerForGUI() { return null; }
        @Override public void setActiveView(String name) {}
        @Override public void showAIPlayedSchwarzMessageDiscarding(String playerName, CardList discardedCards) {}
        @Override public void showAIPlayedSchwarzMessageCardPlay(String playerName, Card card) {}
    }
}
