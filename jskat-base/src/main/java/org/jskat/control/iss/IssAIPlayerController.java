package org.jskat.control.iss;

import org.jskat.ai.ml.MLPlayerPro;
import org.jskat.data.GameAnnouncement;
import org.jskat.data.GameContract;
import org.jskat.data.SkatGameData;
import org.jskat.data.Trick;
import org.jskat.data.iss.MoveInformation;
import org.jskat.data.iss.MoveType;
import org.jskat.util.Card;
import org.jskat.util.CardList;
import org.jskat.util.Player;
import org.jskat.util.SkatConstants;
import org.jskat.util.rule.SkatRule;
import org.jskat.util.rule.SkatRuleFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Controls an AI player on ISS tables.
 * Bridges ISS move updates to AI player decisions and sends responses back to ISS.
 */
public class IssAIPlayerController {

    private static final Logger log = LoggerFactory.getLogger(IssAIPlayerController.class);

    private final IssController issController;
    private final MLPlayerPro aiPlayer;
    private String tableName;
    private Player playerPosition;
    private String loginName;

    // Track which player names correspond to which positions at this table
    private String foreHandName;
    private String middleHandName;
    private String rearHandName;

    // Track cards played in current trick to know when to start new tricks
    private int cardsPlayedInCurrentTrick = 0;
    private int currentTrickNumber = -1;

    // Track the last player who made a bidding move to prevent acting out of turn
    private Player lastBiddingPlayer = null;
    private boolean aiHasActedThisBiddingRound = false;
    private boolean gameAnnounced = false;
    private boolean passSent = false;

    /**
     * Creates a new ISS AI Player Controller
     *
     * @param issController The ISS controller to send moves through
     * @param loginName     The logged-in user's name
     */
    public IssAIPlayerController(final IssController issController, final String loginName) {
        this.issController = issController;
        this.loginName = loginName;
        this.aiPlayer = new MLPlayerPro();
        log.info("IssAIPlayerController created for user: {}", loginName);
    }

    /**
     * Initializes the AI player for a specific table and game.
     *
     * @param tableName Table name
     * @param gameData  Game data containing player positions
     */
    public void initializeForTable(final String tableName, final SkatGameData gameData) {
        this.tableName = tableName;
        this.gameAnnounced = false;
        this.passSent = false;

        // Determine which position the logged-in user has
        foreHandName = gameData.getPlayerName(Player.FOREHAND);
        middleHandName = gameData.getPlayerName(Player.MIDDLEHAND);
        rearHandName = gameData.getPlayerName(Player.REARHAND);

        if (loginName.equals(foreHandName)) {
            playerPosition = Player.FOREHAND;
        } else if (loginName.equals(middleHandName)) {
            playerPosition = Player.MIDDLEHAND;
        } else if (loginName.equals(rearHandName)) {
            playerPosition = Player.REARHAND;
        } else {
            log.warn("Login name {} not found at table {}", loginName, tableName);
            return;
        }

        log.info("AI player seated at table {} as {}", tableName, playerPosition);

        // Initialize AI player
        aiPlayer.setPlayerName(loginName);
        aiPlayer.newGame(playerPosition);
    }

    /**
     * Handles a move update from ISS and potentially triggers an AI decision.
     *
     * @param tableName       Table name
     * @param moveInformation Move information
     * @param gameData        Current game data
     */
    public void onMoveReceived(final String tableName, final MoveInformation moveInformation, final SkatGameData gameData) {
        if (!tableName.equals(this.tableName)) {
            return; // Not our table
        }

        // Synchronize AI player's knowledge with the move
        synchronizePlayerKnowledge(moveInformation, gameData);

        // Check if it's the AI's turn to make a decision
        handlePotentialAITurn(moveInformation, gameData);
    }

    /**
     * Synchronizes the AI player's internal knowledge based on ISS move updates.
     *
     * @param moveInformation Move information from ISS
     * @param gameData        Current game data
     */
    private void synchronizePlayerKnowledge(final MoveInformation moveInformation, final SkatGameData gameData) {
        final MoveType moveType = moveInformation.getType();
        final Player movePlayer = moveInformation.getPlayer();

        switch (moveType) {
            case DEAL:
                // Give the AI player its cards
                final CardList myCards = moveInformation.getCards(playerPosition);
                if (myCards != null && !myCards.isEmpty()) {
                    log.debug("AI player dealt cards: {}", myCards);
                    aiPlayer.takeCards(myCards);
                    aiPlayer.setUpBidding();

                    // Reset bidding tracking for new game
                    lastBiddingPlayer = null;
                    aiHasActedThisBiddingRound = false;
                }
                break;

            case BID:
            case HOLD_BID:
                // Inform AI player of bidding actions
                if (movePlayer != null) {
                    final int bidValue = moveInformation.getBidValue();
                    log.debug("Player {} bid {}", movePlayer, bidValue);
                    aiPlayer.bidByPlayer(movePlayer, bidValue);
                    lastBiddingPlayer = movePlayer;
                    if (movePlayer == playerPosition) {
                        aiHasActedThisBiddingRound = true;
                    } else {
                        // Another player acted, reset our flag so we can respond
                        aiHasActedThisBiddingRound = false;
                    }
                }
                break;

            case PASS:
                // Inform AI player of pass
                if (movePlayer != null) {
                    log.debug("Player {} passed", movePlayer);
                    aiPlayer.bidByPlayer(movePlayer, 0);
                    lastBiddingPlayer = movePlayer;
                    if (movePlayer == playerPosition) {
                        aiHasActedThisBiddingRound = true;
                    } else {
                        // Another player acted, reset our flag so we can respond
                        aiHasActedThisBiddingRound = false;
                    }
                }
                break;

            case PICK_UP_SKAT:
                // Give AI player the skat if it's the declarer
                if (gameData.getDeclarer() == playerPosition) {
                    final CardList skat = moveInformation.getSkat();
                    if (skat != null && !skat.isEmpty()) {
                        log.debug("AI player received skat: {}", skat);
                        aiPlayer.takeSkat(skat);
                    }
                }
                break;

            case GAME_ANNOUNCEMENT:
                // Initialize the AI player with game type and contract
                final GameAnnouncement announcement = moveInformation.getGameAnnouncement();
                if (announcement != null) {
                    final Player declarer = gameData.getDeclarer();
                    log.debug("Game announced: {} by {}", announcement.contract().gameType(), declarer);

                    // Ensure all players have bid information from gameData
                    // This fixes cases where ISS doesn't send all bidding moves
                    for (Player player : Player.values()) {
                        final int playerBid = gameData.getMaxPlayerBid(player);
                        if (playerBid > 0) {
                            log.debug("Syncing bid for {}: {}", player, playerBid);
                            aiPlayer.bidByPlayer(player, playerBid);
                        } else if (gameData.isPlayerPass(player)) {
                            log.debug("Syncing pass for {}", player);
                            aiPlayer.bidByPlayer(player, 0);
                        }
                    }

                    aiPlayer.startGame(declarer, announcement.contract());

                    // Initialize first trick
                    currentTrickNumber = 0;
                    cardsPlayedInCurrentTrick = 0;
                    aiPlayer.newTrick(0, Player.FOREHAND);
                }
                break;

            case CARD_PLAY:
                // Track card plays
                if (movePlayer != null) {
                    final Card card = moveInformation.getCard();
                    log.debug("Player {} played {}", movePlayer, card);

                    // Check if we completed a trick and need to start a new one
                    if (cardsPlayedInCurrentTrick == 3) {
                        // Previous trick completed, start new trick
                        currentTrickNumber++;
                        cardsPlayedInCurrentTrick = 0;

                        final Trick currentTrick = gameData.getCurrentTrick();
                        if (currentTrick != null) {
                            log.debug("Starting trick {}, forehand: {}", currentTrickNumber, currentTrick.getForeHand());
                            aiPlayer.newTrick(currentTrickNumber, currentTrick.getForeHand());
                        }
                    }

                    aiPlayer.cardPlayed(movePlayer, card);
                    cardsPlayedInCurrentTrick++;
                }
                break;

            default:
                // Other move types don't require knowledge updates
                break;
        }
    }

    /**
     * Checks if it's the AI's turn and invokes the appropriate decision method.
     *
     * @param moveInformation Latest move information
     * @param gameData        Current game data
     */
    private void handlePotentialAITurn(final MoveInformation moveInformation, final SkatGameData gameData) {
        final MoveType moveType = moveInformation.getType();
        final Player movePlayer = moveInformation.getPlayer();

        // CRITICAL: Never act on our own moves being echoed back
        // EXCEPTION: If we (or user) picked up skat, we must proceed to announce game
        // EXCEPTION: If we played the last card of a trick and won, we must lead the next trick
        // EXCEPTION: If we announced the game, we might be Forehand and need to lead the first trick
        // EXCEPTION: If we won the bidding (BID/PASS/HOLD), we must proceed to pick up skat or announce
        if (movePlayer == playerPosition
                && moveType != MoveType.PICK_UP_SKAT
                && moveType != MoveType.CARD_PLAY
                && moveType != MoveType.GAME_ANNOUNCEMENT
                && moveType != MoveType.BID
                && moveType != MoveType.PASS
                && moveType != MoveType.HOLD_BID) {
            log.debug("Ignoring our own move echo: {}", moveType);
            return;
        }

        // For bidding, we need special logic based on the current bid state
        if (moveType == MoveType.DEAL || moveType == MoveType.BID || moveType == MoveType.HOLD_BID || moveType == MoveType.PASS) {
            handleBiddingTurn(gameData);
        }
        // SKAT_REQUEST is ISS explicitly asking us to decide on picking up skat
        else if (moveType == MoveType.SKAT_REQUEST && gameData.getDeclarer() == playerPosition) {
            handlePickupDecision(gameData);
        }
        // After picking up skat, declarer needs to discard and announce
        else if (moveType == MoveType.PICK_UP_SKAT && gameData.getDeclarer() == playerPosition) {
            if (moveInformation.getSkat() != null && !moveInformation.getSkat().isEmpty()) {
                handleGameAnnouncement(gameData);
            } else {
                log.warn("Received PICK_UP_SKAT without cards. Waiting for actual skat cards.");
            }
        }
        // During trick playing, check if it's our turn based on the current trick
        else if (moveType == MoveType.CARD_PLAY || moveType == MoveType.GAME_ANNOUNCEMENT) {
            handlePotentialCardPlay(gameData);
        }
    }

    /**
     * Handles bidding decisions during the bidding phase.
     * This is called after each bidding move to check if it's the AI's turn.
     *
     * @param gameData Current game data
     */
    private void handleBiddingTurn(final SkatGameData gameData) {
        // Check if we've already passed
        if (gameData.isPlayerPass(playerPosition) || passSent) {
            return; // Already passed
        }

        // Check if bidding is finished
        if (gameData.getNumberOfPasses() >= 2) {
            // If bidding is finished and we are the declarer, we might need to pick up skat
            // This handles the case where ISS doesn't send an explicit SKAT_REQUEST
            if (gameData.getDeclarer() == playerPosition && !gameData.isSkatPickedUp() && gameData.getGameState() != SkatGameData.GameState.DECLARING) {
                 log.info("Bidding finished, AI is declarer. Triggering pickup decision.");
                 handlePickupDecision(gameData);
                 return;
            }

            // If we are Forehand and everyone else passed, we still need to bid (or pass)
            boolean forehandActive = playerPosition == Player.FOREHAND
                    && !gameData.isPlayerPass(Player.FOREHAND)
                    && gameData.getMaxPlayerBid(Player.FOREHAND) == 0;

            if (!forehandActive) {
                return; // Bidding is over
            }
            log.info("Everyone passed. Forehand (AI) must decide to bid 18 or pass.");
        }

        // Rearhand must wait until one of the other players passes (end of Phase 1)
        if (playerPosition == Player.REARHAND) {
            if (!gameData.isPlayerPass(Player.FOREHAND) && !gameData.isPlayerPass(Player.MIDDLEHAND)) {
                return;
            }
        }

        // Don't act if we just acted (prevent rapid-fire bidding)
        if (aiHasActedThisBiddingRound) {
            log.debug("AI already acted this bidding round, waiting for opponent");
            return;
        }

        // Don't act if the last move was from us (prevent double bidding)
        if (lastBiddingPlayer == playerPosition) {
            log.debug("Last bidding player was us, waiting for opponent");
            return;
        }

        // Determine if it's actually our turn based on bidding protocol
        // If no one has bid yet (all at 0 or passed), and we're middlehand, we should start
        // Otherwise, we should only respond if someone else just bid/held
        if (lastBiddingPlayer == null) {
            // Game just started
            if (playerPosition != Player.MIDDLEHAND) {
                return; // Only middlehand bids first
            }
        } else if (gameData.isPlayerPass(lastBiddingPlayer)) {
            // Handle opponent pass logic:
            // - FOREHAND: Never acts on a pass (waits for R to bid or becomes Declarer)
            // - MIDDLEHAND: Acts if F passed (starts bidding vs R). If R passed, M wins (wait).
            // - REARHAND: Acts if M passed (starts bidding vs F). If F passed, R wins (wait).

            if (playerPosition == Player.FOREHAND) {
                // Normally Forehand waits, unless everyone else passed (then Forehand wins)
                if (gameData.getNumberOfPasses() < 2) {
                    return;
                }
            }
            if (playerPosition == Player.MIDDLEHAND && lastBiddingPlayer == Player.REARHAND) {
                return;
            }
        }

        final int maxBidValue = gameData.getMaxBidValue();
        final int nextBidValue = SkatConstants.getNextBidValue(maxBidValue);

        log.debug("AI's turn to bid. Next bid value: {}", nextBidValue);

        // Ask AI whether to bid (returns the bid value, or 0 to pass)
        final int aiBidValue = aiPlayer.bidMore(nextBidValue);

        // Mark that we're acting
        aiHasActedThisBiddingRound = true;

        if (aiBidValue > 0) {
            // Determine if we're the announcer (bidding a number) or listener (holding)
            // In Skat:
            // - Forehand is ALWAYS the listener (says "yes"/hold or pass)
            // - Middlehand is announcer in phase 1, listener in phase 2 (if forehand passed)
            // - Rearhand is ALWAYS the announcer

            boolean isAnnouncer;
            if (playerPosition == Player.FOREHAND) {
                // Forehand is normally listener, UNLESS everyone else passed.
                isAnnouncer = gameData.getNumberOfPasses() >= 2;
            } else if (playerPosition == Player.REARHAND) {
                isAnnouncer = true; // Rearhand is always announcer
            } else {
                // Middlehand: announcer in phase 1, listener in phase 2
                // If forehand has passed, we're now listener against rearhand
                isAnnouncer = !gameData.isPlayerPass(Player.FOREHAND);
            }

            if (isAnnouncer) {
                // We're the announcer - bid the next value
                log.info("AI player (announcer) bids {}", nextBidValue);
                issController.sendBidMove(tableName);
            } else {
                // We're the listener - hold (say yes to the bid)
                log.info("AI player (listener) holds bid at {}", nextBidValue);
                issController.sendHoldBidMove(tableName);
            }
        } else {
            // Pass
            log.info("AI player passes");
            issController.sendPassBidMove(tableName);
            passSent = true;
        }
    }

    /**
     * Handles the decision to pick up the skat.
     *
     * @param gameData Current game data
     */
    private void handlePickupDecision(final SkatGameData gameData) {
        final boolean shouldPickup = aiPlayer.pickUpSkat();

        if (shouldPickup) {
            log.info("AI player picks up skat");
            gameAnnounced = false; // Reset flag as we will announce later
            issController.sendPickUpSkatMove(tableName);
        } else {
            if (gameAnnounced) {
                log.warn("Game already announced, skipping duplicate announcement.");
                return;
            }
            // Play hand game - send game announcement immediately
            log.info("AI player plays hand game");
            final GameContract contract = aiPlayer.announceGame();
            final GameAnnouncement announcement = new GameAnnouncement(contract); // No discarded cards for hand game
            issController.sendGameAnnouncementMove(tableName, announcement);
            gameAnnounced = true;
        }
    }

    /**
     * Handles game announcement after picking up skat.
     *
     * @param gameData Current game data
     */
    private void handleGameAnnouncement(final SkatGameData gameData) {
        if (gameAnnounced) {
            log.warn("Game already announced, skipping duplicate announcement.");
            return;
        }

        // Get discard from AI
        final CardList discard = aiPlayer.discardSkat();
        log.debug("AI player discards: {}", discard);

        // Get game announcement
        final GameContract contract = aiPlayer.announceGame();
        final GameAnnouncement announcement = new GameAnnouncement(contract, discard);
        log.info("AI player announces: {}", announcement);

        // Manually update gameData to reflect discard immediately (before masked echo)
        gameData.setDiscardedSkat(playerPosition, discard);

        // Send to ISS
        issController.sendGameAnnouncementMove(tableName, announcement);
        gameAnnounced = true;
    }

    /**
     * Handles potential card play during trick playing.
     * Checks if it's the AI's turn based on the current trick state.
     *
     * @param gameData Current game data
     */
    private void handlePotentialCardPlay(final SkatGameData gameData) {
        // Determine if it's our turn based on the current trick
        final Trick currentTrick = gameData.getCurrentTrick();
        if (currentTrick == null) {
            log.warn("Current trick is null, cannot determine turn");
            return;
        }

        // Sync AI to new trick if GameData has advanced
        // This handles cases where we lead the new trick (no move received yet)
        if (currentTrick.getTrickNumberInGame() > currentTrickNumber) {
            log.debug("Syncing AI to new trick {}", currentTrick.getTrickNumberInGame());
            currentTrickNumber = currentTrick.getTrickNumberInGame();
            cardsPlayedInCurrentTrick = 0;
            aiPlayer.newTrick(currentTrickNumber, currentTrick.getForeHand());
        }

        // Figure out whose turn it is based on cards played in the trick
        Player nextPlayer = currentTrick.getForeHand();
        int cardsInTrick = 0;

        if (currentTrick.getFirstCard() != null) {
            nextPlayer = nextPlayer.getLeftNeighbor();
            cardsInTrick++;
        }
        if (currentTrick.getSecondCard() != null) {
            nextPlayer = nextPlayer.getLeftNeighbor();
            cardsInTrick++;
        }
        if (currentTrick.getThirdCard() != null) {
            cardsInTrick++;
        }

        log.debug("Current trick: forehand={}, cardsInTrick={}, nextPlayer={}, aiPosition={}",
                  currentTrick.getForeHand(), cardsInTrick, nextPlayer, playerPosition);

        // Check if trick is completed (3 cards)
        if (currentTrick.getThirdCard() != null) {

            // Safety check: if we already have 10 tricks, the game is over.
            if (gameData.getTricks().size() >= 10) {
                 log.debug("Game over (10 tricks), stopping AI play.");
                 return;
            }

            SkatRule rules = SkatRuleFactory.getSkatRules(gameData.getGameType());
            Player winner = rules.calculateTrickWinner(gameData.getGameType(), currentTrick);

            if (winner == playerPosition && currentTrickNumber < 9) {
                log.info("AI won trick {}, leading next trick.", currentTrickNumber);

                // Advance local state
                currentTrickNumber++;
                cardsPlayedInCurrentTrick = 0;
                aiPlayer.newTrick(currentTrickNumber, playerPosition);

                // Play card
                final Card cardToPlay = safePlayCard();
                if (cardToPlay != null) {
                    log.info("AI player (position={}) leads card: {}", playerPosition, cardToPlay);
                    issController.sendCardMove(tableName, cardToPlay);
                }
                return;
            }
        } else if (nextPlayer == playerPosition && cardsInTrick < 3) {
            // If it's our turn and the trick isn't complete, play a card
            final Card cardToPlay = safePlayCard();
            if (cardToPlay != null) {
                log.info("AI player (position={}) plays card: {}", playerPosition, cardToPlay);
                issController.sendCardMove(tableName, cardToPlay);
            }
        }
    }

    /**
     * Safely attempts to play a card, catching exceptions from empty hand state.
     */
    private Card safePlayCard() {
        try {
            return aiPlayer.playCard();
        } catch (IllegalArgumentException e) {
            log.warn("AI player has no playable cards (hand likely out of sync): {}", e.getMessage());
            return null;
        }
    }

    /**
     * Cleans up after a game ends.
     */
    public void onGameEnd() {
        log.info("Game ended at table {}", tableName);
        // Reset for next game
        cardsPlayedInCurrentTrick = 0;
        currentTrickNumber = -1;
        lastBiddingPlayer = null;
        aiHasActedThisBiddingRound = false;
        gameAnnounced = false;
        passSent = false;
        tableName = null;
        playerPosition = null;
    }

    /**
     * Gets the player position of the AI.
     *
     * @return Player position
     */
    public Player getPlayerPosition() {
        return playerPosition;
    }
}
