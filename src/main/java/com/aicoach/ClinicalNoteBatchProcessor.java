package com.aicoach;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.anthropic.AnthropicChatModel;

import java.io.*;
import java.nio.file.*;
import java.util.List;

public class ClinicalNoteBatchProcessor {

    private static final String SYSTEM_PROMPT = """
            You are a clinical documentation specialist.
            Your job is to extract key facts from raw clinical notes for nurse triage intake.
            
            Always respond using EXACTLY this format — no deviations, no extra commentary:
            
            CHIEF_COMPLAINT: <one sentence>
            SYMPTOMS: <comma-separated list>
            DURATION: <e.g. "3 days", "onset today">
            SEVERITY: <LOW | MEDIUM | HIGH>
            FOLLOW_UP_REQUIRED: <YES | NO>
            SUMMARY: <2-3 sentence plain English summary for a non-clinical reader>
            """;

    public static void main(String[] args) throws Exception {

        AnthropicChatModel model = AnthropicChatModel.builder()
                .apiKey(System.getenv("ANTHROPIC_API_KEY"))
                .modelName("claude-haiku-4-5")
                .temperature(0.0)
                .maxTokens(512)
                .build();

        // Load notes from file
        List<String> notes = Files.readAllLines(
                Path.of("/Users/akhilchowdary/Documents/week01-llm-fundamentals/src/main/resources/clinical_notes")
        );

        // Token tracking
        int totalInputTokens = 0;
        int totalOutputTokens = 0;

        // CSV output
        BufferedWriter csv = new BufferedWriter(new FileWriter("output.csv"));
        csv.write("PatientLine,CHIEF_COMPLAINT,SYMPTOMS,DURATION,SEVERITY,FOLLOW_UP_REQUIRED");
        csv.newLine();

        System.out.println("Processing " + notes.size() + " notes...\n");

        for (int i = 0; i < notes.size(); i++) {
            String note = notes.get(i).trim();
            if (note.isEmpty()) continue;

            System.out.println("--- Note " + (i + 1) + " ---");

            var response = model.chat(
                    List.of(
                            SystemMessage.from(SYSTEM_PROMPT),
                            UserMessage.from("Summarize this clinical note:\n\n" + note)
                    )
            );

            String output = response.aiMessage().text();
            System.out.println(output);

            // Token tracking
            totalInputTokens += response.tokenUsage().inputTokenCount();
            totalOutputTokens += response.tokenUsage().outputTokenCount();

            // Warn if budget exceeded
            if (totalInputTokens > 10_000) {
                System.out.println("⚠️  WARNING: Input token budget exceeded 10,000!");
            }

            // Write to CSV
            csv.write(String.format("%d,%s,%s,%s,%s,%s",
                    i + 1,
                    extractField(output, "CHIEF_COMPLAINT"),
                    extractField(output, "SYMPTOMS"),
                    extractField(output, "DURATION"),
                    extractField(output, "SEVERITY"),
                    extractField(output, "FOLLOW_UP_REQUIRED")
            ));
            csv.newLine();
        }

        csv.close();

        // Final summary
        System.out.println("\n=== SESSION TOKEN USAGE ===");
        System.out.println("Total input tokens:  " + totalInputTokens);
        System.out.println("Total output tokens: " + totalOutputTokens);
        System.out.printf("Total estimated cost: ~$%.5f%n",
                estimateCostUSD(totalInputTokens, totalOutputTokens));
        System.out.println("CSV written to: output.csv");
    }

    private static String extractField(String output, String fieldName) {
        for (String line : output.split("\n")) {
            if (line.startsWith(fieldName + ":")) {
                return line.substring(fieldName.length() + 1).trim()
                        .replace(",", ";"); // escape commas for CSV
            }
        }
        return "NOT_FOUND";
    }

    private static double estimateCostUSD(int inputTokens, int outputTokens) {
        return (inputTokens / 1_000_000.0 * 0.80) + (outputTokens / 1_000_000.0 * 4.00);
    }
}